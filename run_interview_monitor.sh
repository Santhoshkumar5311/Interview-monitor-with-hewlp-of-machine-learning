#!/bin/bash

# Interview Monitor System - Linux/Mac Launcher
# AI-powered interview candidate analysis with computer vision

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored output
print_banner() {
    echo -e "${CYAN}================================================================"
    echo -e "🎯 INTERVIEW MONITOR SYSTEM"
    echo -e "================================================================"
    echo -e "AI-powered interview candidate analysis with computer vision"
    echo -e "===============================================================${NC}"
    echo
}

# Function to show menu
show_menu() {
    echo -e "${YELLOW}What would you like to do?${NC}"
    echo
    echo "1. Run Demo Mode (no camera/microphone required)"
    echo "2. Run Full Interview Monitor (requires camera/microphone)"
    echo "3. Run HUD Interface Only"
    echo "4. Run System Tests"
    echo "5. Show Help"
    echo "6. Exit"
    echo
}

# Function to run demo mode
run_demo() {
    echo
    echo -e "${GREEN}🎭 Starting Demo Mode...${NC}"
    python3 main.py --demo
}

# Function to run full monitor
run_monitor() {
    echo
    echo -e "${GREEN}🎥 Starting Full Interview Monitor...${NC}"
    python3 main.py --monitor
}

# Function to run HUD only
run_hud() {
    echo
    echo -e "${GREEN}🖥️ Starting HUD Interface...${NC}"
    python3 main.py --hud
}

# Function to run tests
run_tests() {
    echo
    echo -e "${GREEN}🧪 Running System Tests...${NC}"
    python3 main.py --test
}

# Function to show help
show_help() {
    echo
    python3 main.py --help
    echo
    read -p "Press Enter to continue..."
}

# Function to handle exit
cleanup() {
    echo
    echo -e "${YELLOW}👋 Goodbye!${NC}"
    echo
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Main loop
while true; do
    print_banner
    show_menu
    
    read -p "Enter your choice (1-6): " choice
    
    case $choice in
        1)
            run_demo
            ;;
        2)
            run_monitor
            ;;
        3)
            run_hud
            ;;
        4)
            run_tests
            ;;
        5)
            show_help
            ;;
        6)
            cleanup
            ;;
        *)
            echo -e "${RED}Invalid choice. Please enter 1-6.${NC}"
            ;;
    esac
    
    echo
    read -p "Press Enter to return to menu..."
    clear
done
