"""
Interview Monitor HUD Interface
PyQt-based interface with live meters, alerts, and timeline heatmap
"""

import sys
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np

try:
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                                QHBoxLayout, QGridLayout, QLabel, QProgressBar, 
                                QTextEdit, QScrollArea, QFrame, QPushButton, 
                                QSlider, QCheckBox, QGroupBox, QSplitter)
    from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QPropertyAnimation, QEasingCurve
    from PyQt5.QtGui import QPainter, QColor, QFont, QPalette, QPixmap, QLinearGradient
    from PyQt5.QtChart import QChart, QChartView, QLineSeries, QValueAxis, QDateTimeAxis
    PYQT5_AVAILABLE = True
except ImportError:
    PYQT5_AVAILABLE = False
    logging.warning("PyQt5 not available. HUD interface will not work.")

logger = logging.getLogger(__name__)

@dataclass
class HUDMetrics:
    """Metrics to display in the HUD"""
    confidence: float
    relevance: float
    sentiment: float
    emotion: str
    toxicity: str
    eye_contact: float
    speaking_rate: float
    pause_ratio: float
    timestamp: datetime

class LiveMeter(QWidget):
    """Custom live meter widget with animated progress"""
    
    def __init__(self, title: str, min_val: float = 0.0, max_val: float = 1.0, 
                 color_scheme: str = "blue", parent=None):
        super().__init__(parent)
        self.title = title
        self.min_val = min_val
        self.max_val = max_val
        self.color_scheme = color_scheme
        self.current_value = 0.0
        self.target_value = 0.0
        
        self.init_ui()
        self.setup_animation()
    
    def init_ui(self):
        """Initialize the UI components"""
        layout = QVBoxLayout()
        
        # Title
        title_label = QLabel(self.title)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("""
            QLabel {
                font-size: 12px;
                font-weight: bold;
                color: #333;
                margin-bottom: 5px;
            }
        """)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1000)  # High precision
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet(self._get_progress_style())
        
        # Value label
        self.value_label = QLabel("0.0")
        self.value_label.setAlignment(Qt.AlignCenter)
        self.value_label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                font-weight: bold;
                color: #555;
            }
        """)
        
        layout.addWidget(title_label)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.value_label)
        
        self.setLayout(layout)
        self.setFixedSize(120, 100)
    
    def setup_animation(self):
        """Setup smooth animation for value changes"""
        self.animation = QPropertyAnimation(self, b"current_value")
        self.animation.setDuration(300)
        self.animation.setEasingCurve(QEasingCurve.OutCubic)
    
    def _get_progress_style(self) -> str:
        """Get progress bar style based on color scheme"""
        colors = {
            "blue": ("#4A90E2", "#2E5BBA"),
            "green": ("#7ED321", "#4A9013"),
            "red": ("#D0021B", "#9B0014"),
            "orange": ("#F5A623", "#D68910"),
            "purple": ("#9013FE", "#6B0FB8")
        }
        
        primary, secondary = colors.get(self.color_scheme, colors["blue"])
        
        return f"""
            QProgressBar {{
                border: 2px solid #E0E0E0;
                border-radius: 10px;
                background-color: #F5F5F5;
                text-align: center;
            }}
            QProgressBar::chunk {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {primary}, stop:1 {secondary});
                border-radius: 8px;
            }}
        """
    
    def set_value(self, value: float, animate: bool = True):
        """Set the meter value with optional animation"""
        self.target_value = max(self.min_val, min(self.max_val, value))
        
        if animate:
            self.animation.setStartValue(self.current_value)
            self.animation.setEndValue(self.target_value)
            self.animation.start()
        else:
            self.current_value = self.target_value
            self._update_display()
    
    def _update_display(self):
        """Update the display with current value"""
        # Update progress bar
        normalized_value = (self.current_value - self.min_val) / (self.max_val - self.min_val)
        self.progress_bar.setValue(int(normalized_value * 1000))
        
        # Update value label
        self.value_label.setText(f"{self.current_value:.2f}")
        
        # Update color based on value
        self._update_color()
    
    def _update_color(self):
        """Update color based on current value"""
        normalized_value = (self.current_value - self.min_val) / (self.max_val - self.min_val)
        
        if normalized_value < 0.3:
            color_scheme = "red"
        elif normalized_value < 0.7:
            color_scheme = "orange"
        else:
            color_scheme = "green"
        
        if color_scheme != self.color_scheme:
            self.color_scheme = color_scheme
            self.progress_bar.setStyleSheet(self._get_progress_style())
    
    def get_current_value(self) -> float:
        """Get current animated value"""
        return self.current_value
    
    def set_current_value(self, value: float):
        """Set current value (for animation)"""
        self.current_value = value
        self._update_display()

class TimelineHeatmap(QWidget):
    """Timeline heatmap widget showing metric history"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.metrics_history: List[HUDMetrics] = []
        self.max_history = 100
        self.cell_size = 20
        self.margin = 10
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize the UI"""
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Timeline Heatmap")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("""
            QLabel {
                font-size: 16px;
                font-weight: bold;
                color: #333;
                margin-bottom: 10px;
            }
        """)
        
        # Scroll area for heatmap
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # Heatmap widget
        self.heatmap_widget = QWidget()
        self.heatmap_widget.setMinimumHeight(200)
        
        scroll_area.setWidget(self.heatmap_widget)
        
        layout.addWidget(title)
        layout.addWidget(scroll_area)
        
        self.setLayout(layout)
    
    def add_metrics(self, metrics: HUDMetrics):
        """Add new metrics to the timeline"""
        self.metrics_history.append(metrics)
        
        # Keep only recent history
        if len(self.metrics_history) > self.max_history:
            self.metrics_history.pop(0)
        
        self.update_heatmap()
    
    def update_heatmap(self):
        """Update the heatmap display"""
        self.heatmap_widget.update()
    
    def paintEvent(self, event):
        """Custom paint event for heatmap"""
        if not self.metrics_history:
            return
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Calculate layout
        metrics_count = len(self.metrics_history)
        rows = 4  # confidence, relevance, sentiment, eye_contact
        cols = min(metrics_count, 25)  # Show max 25 time points
        
        cell_width = max(self.cell_size, (self.width() - 2 * self.margin) // cols)
        cell_height = max(self.cell_size, (self.height() - 2 * self.margin) // rows)
        
        # Draw heatmap cells
        for i, metrics in enumerate(self.metrics_history[-cols:]):
            x = self.margin + i * cell_width
            
            # Confidence row
            y = self.margin
            self._draw_heatmap_cell(painter, x, y, cell_width, cell_height, 
                                  metrics.confidence, "Confidence")
            
            # Relevance row
            y = self.margin + cell_height
            self._draw_heatmap_cell(painter, x, y, cell_width, cell_height, 
                                  metrics.relevance, "Relevance")
            
            # Sentiment row
            y = self.margin + 2 * cell_height
            self._draw_heatmap_cell(painter, x, y, cell_width, cell_height, 
                                  (metrics.sentiment + 1) / 2, "Sentiment")
            
            # Eye contact row
            y = self.margin + 3 * cell_height
            self._draw_heatmap_cell(painter, x, y, cell_width, cell_height, 
                                  metrics.eye_contact, "Eye Contact")
        
        # Draw row labels
        painter.setPen(QColor("#333"))
        painter.setFont(QFont("Arial", 10))
        
        labels = ["Confidence", "Relevance", "Sentiment", "Eye Contact"]
        for i, label in enumerate(labels):
            y = self.margin + i * cell_height + cell_height // 2
            painter.drawText(5, y + 5, label)
    
    def _draw_heatmap_cell(self, painter: QPainter, x: int, y: int, 
                           width: int, height: int, value: float, label: str):
        """Draw a single heatmap cell"""
        # Normalize value to 0-1
        normalized_value = max(0.0, min(1.0, value))
        
        # Color based on value
        if normalized_value < 0.3:
            color = QColor(255, 100, 100)  # Red
        elif normalized_value < 0.7:
            color = QColor(255, 200, 100)  # Orange
        else:
            color = QColor(100, 200, 100)  # Green
        
        # Draw cell
        painter.fillRect(x, y, width, height, color)
        painter.setPen(QColor("#333"))
        painter.drawRect(x, y, width, height)
        
        # Draw value text
        painter.setPen(QColor("#000"))
        painter.setFont(QFont("Arial", 8))
        text = f"{value:.2f}"
        painter.drawText(x, y, width, height, Qt.AlignCenter, text)

class AlertPanel(QWidget):
    """Alert panel for important notifications"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.alerts: List[Tuple[str, str, datetime]] = []  # (type, message, timestamp)
        self.max_alerts = 10
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize the UI"""
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Alerts & Notifications")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("""
            QLabel {
                font-size: 14px;
                font-weight: bold;
                color: #333;
                margin-bottom: 10px;
            }
        """)
        
        # Alerts area
        self.alerts_area = QVBoxLayout()
        
        # Scroll area
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        scroll_widget.setLayout(self.alerts_area)
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setMaximumHeight(200)
        
        layout.addWidget(title)
        layout.addWidget(scroll_area)
        
        self.setLayout(layout)
    
    def add_alert(self, alert_type: str, message: str):
        """Add a new alert"""
        timestamp = datetime.now()
        self.alerts.append((alert_type, message, timestamp))
        
        # Keep only recent alerts
        if len(self.alerts) > self.max_alerts:
            self.alerts.pop(0)
        
        self.update_alerts_display()
    
    def update_alerts_display(self):
        """Update the alerts display"""
        # Clear existing alerts
        for i in reversed(range(self.alerts_area.count())):
            self.alerts_area.itemAt(i).widget().setParent(None)
        
        # Add current alerts
        for alert_type, message, timestamp in self.alerts:
            alert_widget = self._create_alert_widget(alert_type, message, timestamp)
            self.alerts_area.addWidget(alert_widget)
    
    def _create_alert_widget(self, alert_type: str, message: str, timestamp: datetime) -> QWidget:
        """Create an individual alert widget"""
        widget = QFrame()
        widget.setFrameStyle(QFrame.Box)
        widget.setStyleSheet("""
            QFrame {
                background-color: #FFF;
                border: 1px solid #E0E0E0;
                border-radius: 5px;
                padding: 5px;
                margin: 2px;
            }
        """)
        
        layout = QHBoxLayout()
        
        # Alert icon/type
        type_label = QLabel(alert_type.upper())
        type_label.setStyleSheet(f"""
            QLabel {{
                font-size: 10px;
                font-weight: bold;
                color: {'#D0021B' if alert_type == 'warning' else '#F5A623' if alert_type == 'info' else '#7ED321'};
                padding: 2px 6px;
                border-radius: 3px;
                background-color: {'#FFE6E6' if alert_type == 'warning' else '#FFF8E6' if alert_type == 'info' else '#E6F7E6'};
            }}
        """)
        
        # Message
        message_label = QLabel(message)
        message_label.setWordWrap(True)
        message_label.setStyleSheet("""
            QLabel {
                font-size: 11px;
                color: #333;
            }
        """)
        
        # Timestamp
        time_label = QLabel(timestamp.strftime("%H:%M:%S"))
        time_label.setStyleSheet("""
            QLabel {
                font-size: 9px;
                color: #999;
            }
        """)
        
        layout.addWidget(type_label)
        layout.addWidget(message_label, 1)
        layout.addWidget(time_label)
        
        widget.setLayout(layout)
        return widget

class InterviewHUD(QMainWindow):
    """Main Interview Monitor HUD Window"""
    
    def __init__(self):
        super().__init__()
        self.metrics_history: List[HUDMetrics] = []
        self.update_timer = QTimer()
        
        self.init_ui()
        self.setup_timers()
    
    def init_ui(self):
        """Initialize the main UI"""
        self.setWindowTitle("Interview Monitor HUD")
        self.setGeometry(100, 100, 1200, 800)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout()
        
        # Left panel - Live meters
        left_panel = self._create_left_panel()
        
        # Right panel - Timeline and alerts
        right_panel = self._create_right_panel()
        
        # Splitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([600, 600])
        
        main_layout.addWidget(splitter)
        central_widget.setLayout(main_layout)
        
        # Style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #F5F5F5;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #E0E0E0;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
    
    def _create_left_panel(self) -> QWidget:
        """Create the left panel with live meters"""
        panel = QWidget()
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Live Interview Metrics")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("""
            QLabel {
                font-size: 18px;
                font-weight: bold;
                color: #333;
                margin-bottom: 20px;
            }
        """)
        
        # Confidence metrics group
        confidence_group = QGroupBox("Confidence & Relevance")
        confidence_layout = QGridLayout()
        
        self.confidence_meter = LiveMeter("Confidence", 0.0, 1.0, "green")
        self.relevance_meter = LiveMeter("Relevance", 0.0, 1.0, "blue")
        
        confidence_layout.addWidget(self.confidence_meter, 0, 0)
        confidence_layout.addWidget(self.relevance_meter, 0, 1)
        confidence_group.setLayout(confidence_layout)
        
        # Sentiment metrics group
        sentiment_group = QGroupBox("Sentiment & Emotion")
        sentiment_layout = QGridLayout()
        
        self.sentiment_meter = LiveMeter("Sentiment", -1.0, 1.0, "purple")
        self.emotion_label = QLabel("Neutral")
        self.emotion_label.setAlignment(Qt.AlignCenter)
        self.emotion_label.setStyleSheet("""
            QLabel {
                font-size: 16px;
                font-weight: bold;
                color: #333;
                padding: 10px;
                background-color: #E6F7E6;
                border-radius: 5px;
            }
        """)
        
        sentiment_layout.addWidget(self.sentiment_meter, 0, 0)
        sentiment_layout.addWidget(self.emotion_label, 0, 1)
        sentiment_group.setLayout(sentiment_layout)
        
        # Behavioral metrics group
        behavioral_group = QGroupBox("Behavioral Metrics")
        behavioral_layout = QGridLayout()
        
        self.eye_contact_meter = LiveMeter("Eye Contact", 0.0, 1.0, "orange")
        self.speaking_rate_meter = LiveMeter("Speaking Rate", 0.0, 5.0, "blue")
        self.pause_ratio_meter = LiveMeter("Pause Ratio", 0.0, 1.0, "red")
        
        behavioral_layout.addWidget(self.eye_contact_meter, 0, 0)
        behavioral_layout.addWidget(self.speaking_rate_meter, 0, 1)
        behavioral_layout.addWidget(self.pause_ratio_meter, 1, 0)
        behavioral_group.setLayout(behavioral_layout)
        
        # Add all groups
        layout.addWidget(title)
        layout.addWidget(confidence_group)
        layout.addWidget(sentiment_group)
        layout.addWidget(behavioral_group)
        layout.addStretch()
        
        panel.setLayout(layout)
        return panel
    
    def _create_right_panel(self) -> QWidget:
        """Create the right panel with timeline and alerts"""
        panel = QWidget()
        layout = QVBoxLayout()
        
        # Timeline heatmap
        self.timeline_heatmap = TimelineHeatmap()
        
        # Alert panel
        self.alert_panel = AlertPanel()
        
        # Add widgets
        layout.addWidget(self.timeline_heatmap)
        layout.addWidget(self.alert_panel)
        
        panel.setLayout(layout)
        return panel
    
    def setup_timers(self):
        """Setup update timers"""
        self.update_timer.timeout.connect(self.update_display)
        self.update_timer.start(100)  # Update every 100ms
    
    def update_metrics(self, metrics: HUDMetrics):
        """Update the HUD with new metrics"""
        # Update meters
        self.confidence_meter.set_value(metrics.confidence)
        self.relevance_meter.set_value(metrics.relevance)
        self.sentiment_meter.set_value(metrics.sentiment)
        self.eye_contact_meter.set_value(metrics.eye_contact)
        self.speaking_rate_meter.set_value(metrics.speaking_rate)
        self.pause_ratio_meter.set_value(metrics.pause_ratio)
        
        # Update emotion label
        self.emotion_label.setText(metrics.emotion.title())
        
        # Update timeline
        self.timeline_heatmap.add_metrics(metrics)
        
        # Check for alerts
        self._check_alerts(metrics)
        
        # Store metrics
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > 1000:
            self.metrics_history.pop(0)
    
    def _check_alerts(self, metrics: HUDMetrics):
        """Check metrics for alert conditions"""
        # Low confidence alert
        if metrics.confidence < 0.3:
            self.alert_panel.add_alert("warning", f"Low confidence detected: {metrics.confidence:.2f}")
        
        # High toxicity alert
        if metrics.toxicity == "high":
            self.alert_panel.add_alert("warning", "High toxicity detected in speech")
        
        # Poor eye contact alert
        if metrics.eye_contact < 0.2:
            self.alert_panel.add_alert("info", f"Low eye contact: {metrics.eye_contact:.2f}")
        
        # High pause ratio alert
        if metrics.pause_ratio > 0.5:
            self.alert_panel.add_alert("info", f"High pause ratio: {metrics.pause_ratio:.2f}")
    
    def update_display(self):
        """Update display (called by timer)"""
        # This can be used for real-time updates if needed
        pass
    
    def closeEvent(self, event):
        """Handle window close event"""
        self.update_timer.stop()
        event.accept()

def main():
    """Main function to run the HUD"""
    if not PYQT5_AVAILABLE:
        print("PyQt5 is required to run the HUD interface.")
        print("Install with: pip install PyQt5 PyQtChart")
        return
    
    app = QApplication(sys.argv)
    
    # Create and show HUD
    hud = InterviewHUD()
    hud.show()
    
    # Run the application
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
