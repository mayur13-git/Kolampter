# Kolam Scanner Web Application

A web-based application that allows users to capture or upload Kolam designs using their phone camera and analyze them to identify mathematical principles and design patterns.

## 🌟 Features

### 📱 Mobile-Friendly Interface
- **Camera Capture**: Use your phone's camera to capture Kolam designs in real-time
- **Image Upload**: Upload existing Kolam images from your device
- **Responsive Design**: Works perfectly on mobile devices and desktop

### 🔍 Advanced Analysis
- **Symmetry Detection**: Identifies radial, bilateral, and rotational symmetry
- **Fractal Analysis**: Measures fractal dimension and self-similarity
- **Topology Analysis**: Analyzes connected components, holes, and loops
- **Pattern Classification**: Automatically categorizes pattern types

### 📊 Rich Visualizations
- **2D Analysis**: Detailed pattern analysis with overlays
- **3D Visualization**: Three-dimensional representation of patterns
- **Symmetry Visualization**: Visual representation of symmetry properties
- **Fractal Analysis**: Detailed fractal property visualization

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the Web Application
```bash
python run_web_app.py
```

### 3. Access the Application
- The application will automatically open in your browser
- Or manually go to: `http://localhost:5000`

## 📱 How to Use

### Camera Capture
1. Click "Start Camera" to activate your device's camera
2. Point the camera at a Kolam design
3. Ensure good lighting and steady positioning
4. Click "Capture" to take a photo
5. View the analysis results

### Image Upload
1. Click "Upload Image" on the home page
2. Select an image file from your device
3. Or drag and drop an image file
4. Wait for the analysis to complete
5. View the detailed results

## 🔧 Technical Details

### Backend (Flask)
- **Image Processing**: OpenCV for image preprocessing
- **Analysis Engine**: Integrated Kolam analysis system
- **API Endpoints**: RESTful API for image upload and analysis
- **File Management**: Automatic session management and result storage

### Frontend (HTML/CSS/JavaScript)
- **Responsive Design**: Bootstrap 5 for mobile-friendly interface
- **Camera API**: WebRTC for camera access
- **File Upload**: Drag-and-drop file upload support
- **Real-time Updates**: Dynamic result display

### Analysis Pipeline
1. **Image Preprocessing**: Convert to grayscale, resize, enhance contrast
2. **Pattern Analysis**: Apply mathematical analysis algorithms
3. **Visualization Generation**: Create analysis visualizations
4. **Result Compilation**: Generate comprehensive analysis report

## 📊 Analysis Results

The application provides detailed analysis including:

### Pattern Classification
- **Primary Type**: fractal, spiral, geometric, traditional, grid
- **Complexity Level**: low, medium, high
- **Mathematical Properties**: List of identified properties

### Symmetry Analysis
- **Radial Symmetry**: Score and center point detection
- **Bilateral Symmetry**: Horizontal, vertical, and diagonal symmetry
- **Rotational Symmetry**: Best rotational angle and score

### Fractal Properties
- **Box Dimension**: Fractal dimension measurement
- **Self-Similarity**: Self-similarity score
- **Recursive Structure**: Grid periodicity and spiral detection

### Topology
- **Connected Components**: Number of separate elements
- **Holes**: Enclosed regions within the pattern
- **Loops**: Closed curves in the design

## 🎨 Visualizations

### Generated Images
- **Original Image**: The captured or uploaded image
- **Analysis Heatmap**: Overview of all analysis results
- **Symmetry Visualization**: Symmetry lines and center points
- **Fractal Analysis**: Detailed fractal property visualization
- **3D Visualization**: Three-dimensional pattern representation

## 📁 File Structure

```
Kolampter/
├── app.py                 # Flask web application
├── run_web_app.py        # Application launcher
├── templates/            # HTML templates
│   ├── base.html         # Base template
│   ├── index.html        # Home page
│   └── scan.html         # Camera scan page
├── uploads/              # Uploaded images (auto-created)
├── results/              # Analysis results (auto-created)
└── kolam_*.py           # Core analysis modules
```

## 🔒 Security & Privacy

- **Local Processing**: All analysis is performed locally
- **No Data Storage**: Images are processed and discarded
- **Session Management**: Temporary result storage with automatic cleanup
- **File Size Limits**: 16MB maximum file size

## 🌐 Browser Compatibility

### Supported Browsers
- **Chrome**: Full support including camera access
- **Firefox**: Full support including camera access
- **Safari**: Full support including camera access
- **Edge**: Full support including camera access

### Mobile Support
- **iOS Safari**: Full camera and upload support
- **Android Chrome**: Full camera and upload support
- **Responsive Design**: Optimized for all screen sizes

## 🚀 Deployment

### Local Development
```bash
python run_web_app.py
```

### Production Deployment
```bash
# Using Gunicorn (recommended)
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app

# Using Flask's built-in server (development only)
python app.py
```

### Environment Variables
- `FLASK_ENV`: Set to 'production' for production deployment
- `FLASK_DEBUG`: Set to 'False' for production

## 🔧 Configuration

### Camera Settings
- **Resolution**: Automatically optimized for device capabilities
- **Facing Mode**: Uses back camera on mobile devices
- **Quality**: JPEG compression at 80% quality

### Analysis Settings
- **Image Size**: Automatically resized to 1000px max dimension
- **Preprocessing**: Automatic contrast enhancement and binarization
- **Analysis Depth**: Comprehensive analysis with all available metrics

## 📱 Mobile Optimization

### Camera Features
- **Auto-focus**: Automatic focus adjustment
- **Flash Control**: Automatic flash management
- **Orientation**: Automatic orientation detection
- **Stabilization**: Image stabilization for better results

### Touch Interface
- **Touch-friendly**: Large buttons and touch targets
- **Gesture Support**: Swipe and pinch gestures
- **Responsive Layout**: Adapts to all screen sizes

## 🎯 Use Cases

### Educational
- **Art Classes**: Analyze traditional art patterns
- **Mathematics**: Study fractal and geometric properties
- **Cultural Studies**: Explore traditional design principles

### Research
- **Pattern Recognition**: Study mathematical patterns in art
- **Cultural Analysis**: Analyze traditional design evolution
- **Algorithm Development**: Test pattern recognition algorithms

### Personal
- **Art Appreciation**: Understand the mathematics behind art
- **Design Inspiration**: Generate new patterns based on analysis
- **Cultural Exploration**: Learn about traditional art forms

## 🔮 Future Enhancements

### Planned Features
- **Pattern Generation**: Generate new patterns based on analysis
- **Comparison Tools**: Compare multiple patterns
- **Export Options**: Export analysis results and visualizations
- **Social Features**: Share and discuss patterns
- **Database Integration**: Store and search pattern collections

### Advanced Analysis
- **Machine Learning**: AI-powered pattern recognition
- **Historical Analysis**: Compare with historical patterns
- **Cultural Context**: Add cultural and historical information
- **3D Reconstruction**: Create 3D models from 2D patterns

## 🆘 Troubleshooting

### Common Issues

#### Camera Not Working
- **Check Permissions**: Ensure camera access is allowed
- **HTTPS Required**: Some browsers require HTTPS for camera access
- **Browser Compatibility**: Try a different browser

#### Upload Issues
- **File Size**: Ensure file is under 16MB
- **File Format**: Use JPG, PNG, or GIF formats
- **Network**: Check internet connection

#### Analysis Errors
- **Image Quality**: Ensure good image quality and contrast
- **File Format**: Use supported image formats
- **Server Resources**: Check server memory and processing power

### Support
For technical support or feature requests, please check the main project documentation or create an issue in the project repository.

---

**Status**: ✅ **READY FOR USE**

**Access**: http://localhost:5000 (after running the application)  
**Mobile Support**: ✅ Full mobile camera and upload support  
**Browser Support**: ✅ All modern browsers  
**Analysis Features**: ✅ Complete mathematical analysis suite
