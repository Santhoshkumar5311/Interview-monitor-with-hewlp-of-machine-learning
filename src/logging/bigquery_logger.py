"""
BigQuery Logger Module
Logs interview monitoring features to BigQuery with session tracking
"""

import os
import logging
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import uuid

try:
    from google.cloud import bigquery
    from google.oauth2 import service_account
    BIGQUERY_AVAILABLE = True
except ImportError:
    BIGQUERY_AVAILABLE = False
    logging.warning("BigQuery not available. Logging will be disabled.")

logger = logging.getLogger(__name__)

@dataclass
class SessionInfo:
    """Session information for tracking"""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    user_id: Optional[str] = None
    interview_type: Optional[str] = None
    duration_minutes: Optional[float] = None
    status: str = "active"  # active, completed, interrupted

@dataclass
class FeatureLog:
    """Feature data to log"""
    session_id: str
    t_start: datetime
    t_end: datetime
    feature_type: str  # facial, speech, sentiment, fusion
    feature_data: Dict[str, Any]
    confidence: float
    quality_score: float

class BigQueryLogger:
    """BigQuery logger for interview monitoring data"""
    
    def __init__(self, 
                 project_id: Optional[str] = None,
                 dataset_id: str = "interview_monitoring",
                 credentials_path: Optional[str] = None):
        """
        Initialize BigQuery Logger
        
        Args:
            project_id: GCP project ID
            dataset_id: BigQuery dataset ID
            credentials_path: Path to service account JSON
        """
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.credentials_path = credentials_path
        
        if not BIGQUERY_AVAILABLE:
            logger.warning("BigQuery not available. Logging will be disabled.")
            self.client = None
            return
        
        # Initialize BigQuery client
        try:
            if credentials_path and os.path.exists(credentials_path):
                credentials = service_account.Credentials.from_service_account_file(credentials_path)
                self.client = bigquery.Client(credentials=credentials, project=project_id)
            else:
                # Use default credentials
                self.client = bigquery.Client(project=project_id)
            
            logger.info(f"BigQuery client initialized for project: {project_id}")
            
            # Ensure dataset and tables exist
            self._ensure_dataset_and_tables()
            
        except Exception as e:
            logger.error(f"Failed to initialize BigQuery client: {e}")
            self.client = None
    
    def _ensure_dataset_and_tables(self):
        """Ensure the dataset and required tables exist"""
        try:
            # Create dataset if it doesn't exist
            dataset_ref = self.client.dataset(self.dataset_id)
            try:
                self.client.get_dataset(dataset_ref)
                logger.info(f"Dataset {self.dataset_id} already exists")
            except Exception:
                dataset = bigquery.Dataset(dataset_ref)
                dataset.location = "US"  # Set location
                self.client.create_dataset(dataset, timeout=30)
                logger.info(f"Created dataset {self.dataset_id}")
            
            # Create tables if they don't exist
            self._create_sessions_table()
            self._create_features_table()
            
        except Exception as e:
            logger.error(f"Failed to ensure dataset and tables: {e}")
    
    def _create_sessions_table(self):
        """Create sessions table if it doesn't exist"""
        try:
            table_id = f"{self.project_id}.{self.dataset_id}.sessions"
            
            schema = [
                bigquery.SchemaField("session_id", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("start_time", "TIMESTAMP", mode="REQUIRED"),
                bigquery.SchemaField("end_time", "TIMESTAMP", mode="NULLABLE"),
                bigquery.SchemaField("user_id", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("interview_type", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("duration_minutes", "FLOAT64", mode="NULLABLE"),
                bigquery.SchemaField("status", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED"),
                bigquery.SchemaField("updated_at", "TIMESTAMP", mode="REQUIRED")
            ]
            
            table = bigquery.Table(table_id, schema=schema)
            
            # Set clustering and partitioning
            table.clustering_fields = ["user_id", "status"]
            table.time_partitioning = bigquery.TimePartitioning(
                type_=bigquery.TimePartitioningType.DAY,
                field="start_time"
            )
            
            try:
                self.client.get_table(table_id)
                logger.info("Sessions table already exists")
            except Exception:
                self.client.create_table(table)
                logger.info("Created sessions table")
                
        except Exception as e:
            logger.error(f"Failed to create sessions table: {e}")
    
    def _create_features_table(self):
        """Create features table if it doesn't exist"""
        try:
            table_id = f"{self.project_id}.{self.dataset_id}.features"
            
            schema = [
                bigquery.SchemaField("session_id", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("t_start", "TIMESTAMP", mode="REQUIRED"),
                bigquery.SchemaField("t_end", "TIMESTAMP", mode="REQUIRED"),
                bigquery.SchemaField("feature_type", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("feature_data", "JSON", mode="REQUIRED"),
                bigquery.SchemaField("confidence", "FLOAT64", mode="REQUIRED"),
                bigquery.SchemaField("quality_score", "FLOAT64", mode="REQUIRED"),
                bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED")
            ]
            
            table = bigquery.Table(table_id, schema=schema)
            
            # Set clustering and partitioning
            table.clustering_fields = ["session_id", "feature_type"]
            table.time_partitioning = bigquery.TimePartitioning(
                type_=bigquery.TimePartitioningType.DAY,
                field="t_start"
            )
            
            try:
                self.client.get_table(table_id)
                logger.info("Features table already exists")
            except Exception:
                self.client.create_table(table)
                logger.info("Created features table")
                
        except Exception as e:
            logger.error(f"Failed to create features table: {e}")
    
    def start_session(self, 
                     user_id: Optional[str] = None,
                     interview_type: Optional[str] = None) -> SessionInfo:
        """
        Start a new interview session
        
        Args:
            user_id: Optional user identifier
            interview_type: Type of interview
            
        Returns:
            SessionInfo object
        """
        session_id = str(uuid.uuid4())
        start_time = datetime.utcnow()
        
        session_info = SessionInfo(
            session_id=session_id,
            start_time=start_time,
            user_id=user_id,
            interview_type=interview_type
        )
        
        # Log session start to BigQuery
        if self.client:
            try:
                self._insert_session(session_info)
                logger.info(f"Started session: {session_id}")
            except Exception as e:
                logger.error(f"Failed to log session start: {e}")
        
        return session_info
    
    def end_session(self, session_info: SessionInfo, status: str = "completed"):
        """
        End an interview session
        
        Args:
            session_info: Session info to end
            status: Final status of the session
        """
        session_info.end_time = datetime.utcnow()
        session_info.status = status
        
        if session_info.start_time and session_info.end_time:
            duration = (session_info.end_time - session_info.start_time).total_seconds() / 60
            session_info.duration_minutes = duration
        
        # Update session in BigQuery
        if self.client:
            try:
                self._update_session(session_info)
                logger.info(f"Ended session: {session_info.session_id} (Status: {status})")
            except Exception as e:
                logger.error(f"Failed to log session end: {e}")
    
    def log_features(self, 
                    session_id: str,
                    t_start: datetime,
                    t_end: datetime,
                    feature_type: str,
                    feature_data: Dict[str, Any],
                    confidence: float,
                    quality_score: float):
        """
        Log feature data to BigQuery
        
        Args:
            session_id: Session identifier
            t_start: Start timestamp
            t_end: End timestamp
            feature_type: Type of feature
            feature_data: Feature data dictionary
            confidence: Confidence score
            quality_score: Quality score
        """
        if not self.client:
            logger.warning("BigQuery client not available. Feature logging skipped.")
            return
        
        try:
            feature_log = FeatureLog(
                session_id=session_id,
                t_start=t_start,
                t_end=t_end,
                feature_type=feature_type,
                feature_data=feature_data,
                confidence=confidence,
                quality_score=quality_score
            )
            
            self._insert_feature(feature_log)
            logger.debug(f"Logged {feature_type} features for session {session_id}")
            
        except Exception as e:
            logger.error(f"Failed to log features: {e}")
    
    def log_facial_features(self, 
                           session_id: str,
                           t_start: datetime,
                           t_end: datetime,
                           facial_metrics: Dict[str, Any],
                           confidence: float,
                           quality_score: float):
        """Log facial analysis features"""
        self.log_features(
            session_id=session_id,
            t_start=t_start,
            t_end=t_end,
            feature_type="facial",
            feature_data=facial_metrics,
            confidence=confidence,
            quality_score=quality_score
        )
    
    def log_speech_features(self, 
                           session_id: str,
                           t_start: datetime,
                           t_end: datetime,
                           speech_metrics: Dict[str, Any],
                           confidence: float,
                           quality_score: float):
        """Log speech analysis features"""
        self.log_features(
            session_id=session_id,
            t_start=t_start,
            t_end=t_end,
            feature_type="speech",
            feature_data=speech_metrics,
            confidence=confidence,
            quality_score=quality_score
        )
    
    def log_sentiment_features(self, 
                              session_id: str,
                              t_start: datetime,
                              t_end: datetime,
                              sentiment_metrics: Dict[str, Any],
                              confidence: float,
                              quality_score: float):
        """Log sentiment analysis features"""
        self.log_features(
            session_id=session_id,
            t_start=t_start,
            t_end=t_end,
            feature_type="sentiment",
            feature_data=sentiment_metrics,
            confidence=confidence,
            quality_score=quality_score
        )
    
    def log_fusion_features(self, 
                           session_id: str,
                           t_start: datetime,
                           t_end: datetime,
                           fusion_metrics: Dict[str, Any],
                           confidence: float,
                           quality_score: float):
        """Log fusion analysis features"""
        self.log_features(
            session_id=session_id,
            t_start=t_start,
            t_end=t_end,
            feature_type="fusion",
            feature_data=fusion_metrics,
            confidence=confidence,
            quality_score=quality_score
        )
    
    def _insert_session(self, session_info: SessionInfo):
        """Insert session record into BigQuery"""
        table_id = f"{self.project_id}.{self.dataset_id}.sessions"
        
        rows_to_insert = [{
            "session_id": session_info.session_id,
            "start_time": session_info.start_time.isoformat(),
            "end_time": session_info.end_time.isoformat() if session_info.end_time else None,
            "user_id": session_info.user_id,
            "interview_type": session_info.interview_type,
            "duration_minutes": session_info.duration_minutes,
            "status": session_info.status,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }]
        
        errors = self.client.insert_rows_json(table_id, rows_to_insert)
        if errors:
            raise Exception(f"Failed to insert session: {errors}")
    
    def _update_session(self, session_info: SessionInfo):
        """Update session record in BigQuery"""
        table_id = f"{self.project_id}.{self.dataset_id}.sessions"
        
        query = f"""
        UPDATE `{table_id}`
        SET 
            end_time = @end_time,
            duration_minutes = @duration_minutes,
            status = @status,
            updated_at = @updated_at
        WHERE session_id = @session_id
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("end_time", "TIMESTAMP", session_info.end_time),
                bigquery.ScalarQueryParameter("duration_minutes", "FLOAT64", session_info.duration_minutes),
                bigquery.ScalarQueryParameter("status", "STRING", session_info.status),
                bigquery.ScalarQueryParameter("updated_at", "TIMESTAMP", datetime.utcnow()),
                bigquery.ScalarQueryParameter("session_id", "STRING", session_info.session_id)
            ]
        )
        
        query_job = self.client.query(query, job_config=job_config)
        query_job.result()  # Wait for the query to complete
    
    def _insert_feature(self, feature_log: FeatureLog):
        """Insert feature record into BigQuery"""
        table_id = f"{self.project_id}.{self.dataset_id}.features"
        
        rows_to_insert = [{
            "session_id": feature_log.session_id,
            "t_start": feature_log.t_start.isoformat(),
            "t_end": feature_log.t_end.isoformat(),
            "feature_type": feature_log.feature_type,
            "feature_data": json.dumps(feature_log.feature_data),
            "confidence": feature_log.confidence,
            "quality_score": feature_log.quality_score,
            "created_at": datetime.utcnow().isoformat()
        }]
        
        errors = self.client.insert_rows_json(table_id, rows_to_insert)
        if errors:
            raise Exception(f"Failed to insert feature: {errors}")
    
    def get_session_features(self, 
                           session_id: str,
                           feature_type: Optional[str] = None,
                           limit: int = 1000) -> List[Dict[str, Any]]:
        """
        Retrieve features for a specific session
        
        Args:
            session_id: Session identifier
            feature_type: Optional feature type filter
            limit: Maximum number of records to return
            
        Returns:
            List of feature records
        """
        if not self.client:
            logger.warning("BigQuery client not available. Cannot retrieve features.")
            return []
        
        try:
            table_id = f"{self.project_id}.{self.dataset_id}.features"
            
            query = f"""
            SELECT *
            FROM `{table_id}`
            WHERE session_id = @session_id
            """
            
            if feature_type:
                query += " AND feature_type = @feature_type"
            
            query += " ORDER BY t_start DESC LIMIT @limit"
            
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("session_id", "STRING", session_id),
                    bigquery.ScalarQueryParameter("feature_type", "STRING", feature_type) if feature_type else None,
                    bigquery.ScalarQueryParameter("limit", "INT64", limit)
                ]
            )
            
            # Remove None parameters
            job_config.query_parameters = [p for p in job_config.query_parameters if p is not None]
            
            query_job = self.client.query(query, job_config=job_config)
            results = query_job.result()
            
            features = []
            for row in results:
                feature_dict = dict(row)
                # Parse JSON feature data
                if 'feature_data' in feature_dict:
                    feature_dict['feature_data'] = json.loads(feature_dict['feature_data'])
                features.append(feature_dict)
            
            return features
            
        except Exception as e:
            logger.error(f"Failed to retrieve session features: {e}")
            return []
    
    def get_session_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get summary statistics for a session
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session summary dictionary
        """
        if not self.client:
            return None
        
        try:
            features_table = f"{self.project_id}.{self.dataset_id}.features"
            sessions_table = f"{self.project_id}.{self.dataset_id}.sessions"
            
            query = f"""
            SELECT 
                s.session_id,
                s.start_time,
                s.end_time,
                s.duration_minutes,
                s.status,
                COUNT(f.feature_type) as total_features,
                AVG(f.confidence) as avg_confidence,
                AVG(f.quality_score) as avg_quality,
                COUNT(DISTINCT f.feature_type) as feature_types
            FROM `{sessions_table}` s
            LEFT JOIN `{features_table}` f ON s.session_id = f.session_id
            WHERE s.session_id = @session_id
            GROUP BY s.session_id, s.start_time, s.end_time, s.duration_minutes, s.status
            """
            
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("session_id", "STRING", session_id)
                ]
            )
            
            query_job = self.client.query(query, job_config=job_config)
            results = list(query_job.result())
            
            if results:
                return dict(results[0])
            return None
            
        except Exception as e:
            logger.error(f"Failed to get session summary: {e}")
            return None
    
    def cleanup(self):
        """Clean up BigQuery client"""
        if self.client:
            self.client.close()
            logger.info("BigQuery client closed")

class LocalLogger:
    """Fallback local logger when BigQuery is not available"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.sessions_file = os.path.join(log_dir, "sessions.jsonl")
        self.features_file = os.path.join(log_dir, "features.jsonl")
        
        logger.info(f"Local logger initialized in {log_dir}")
    
    def start_session(self, 
                     user_id: Optional[str] = None,
                     interview_type: Optional[str] = None) -> SessionInfo:
        """Start session with local logging"""
        import uuid
        
        session_info = SessionInfo(
            session_id=str(uuid.uuid4()),
            start_time=datetime.utcnow(),
            user_id=user_id,
            interview_type=interview_type
        )
        
        # Log to local file
        with open(self.sessions_file, 'a') as f:
            f.write(json.dumps(asdict(session_info), default=str) + '\n')
        
        return session_info
    
    def end_session(self, session_info: SessionInfo, status: str = "completed"):
        """End session with local logging"""
        session_info.end_time = datetime.utcnow()
        session_info.status = status
        
        # Update local file (simplified - in practice you'd want a proper database)
        logger.info(f"Session ended: {session_info.session_id} (Status: {status})")
    
    def log_features(self, **kwargs):
        """Log features locally"""
        feature_log = FeatureLog(**kwargs)
        
        with open(self.features_file, 'a') as f:
            f.write(json.dumps(asdict(feature_log), default=str) + '\n')
    
    def cleanup(self):
        """Cleanup local logger"""
        pass

def create_logger(use_bigquery: bool = True, **kwargs) -> BigQueryLogger:
    """
    Factory function to create appropriate logger
    
    Args:
        use_bigquery: Whether to use BigQuery
        **kwargs: Logger configuration
        
    Returns:
        Logger instance
    """
    if use_bigquery and BIGQUERY_AVAILABLE:
        return BigQueryLogger(**kwargs)
    else:
        logger.warning("Using local logger (BigQuery not available)")
        return LocalLogger(**kwargs)
