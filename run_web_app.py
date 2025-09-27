"""
Kolam Scanner Web Application Launcher
======================================

Simple launcher script for the Kolam Scanner web application.
"""

import os
import sys
import webbrowser
import time
from threading import Timer

def open_browser():
    """Open the web browser after a short delay."""
    time.sleep(1.5)
    webbrowser.open('http://localhost:5000')

def main():
    """Launch the Kolam Scanner web application."""
    print("🎨 Kolam Scanner Web Application")
    print("=" * 40)
    print("Starting the web server...")
    print("The application will open in your browser automatically.")
    print("If it doesn't open, go to: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    print("=" * 40)
    
    # Open browser in a separate thread
    Timer(1.5, open_browser).start()
    
    # Import and run the Flask app
    try:
        from app import app
        app.run(debug=False, host='0.0.0.0', port=5000)
    except ImportError as e:
        print(f"Error: {e}")
        print("Please make sure all dependencies are installed:")
        print("pip install -r requirements.txt")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nShutting down the server...")
        sys.exit(0)

if __name__ == "__main__":
    main()
