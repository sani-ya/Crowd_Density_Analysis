const { useState, useEffect, useRef } = React;

function App() {
    const [status, setStatus] = useState("Disconnected");
    const [viewMode, setViewMode] = useState("feed"); // 'feed' or 'heatmap'
    const [crowdCount, setCrowdCount] = useState(0);
    const [anomalyDetected, setAnomalyDetected] = useState(false);
    const [densityLevel, setDensityLevel] = useState("Normal"); // Normal, Crowded, Critical
    const [imagePreview, setImagePreview] = useState(null);
    const [isLive, setIsLive] = useState(false);
    
    const fileInputRef = useRef(null);

    // Simulated WebSocket/Backend connection status
    useEffect(() => {
        setTimeout(() => {
            setStatus("Ready to Analyze");
        }, 1500);
    }, []);

    // Helper to determine status class
    const getStatusClass = (count) => {
        if (count < 50) return "status-normal";
        if (count < 150) return "status-crowded";
        return "status-critical";
    };

    const handleUploadClick = () => {
        fileInputRef.current.click();
    };

    const handleFileChange = (e) => {
        const file = e.target.files[0];
        if (file) {
            setIsLive(false);
            const reader = new FileReader();
            reader.onloadend = () => {
                setImagePreview(reader.result);
                setStatus("Analyzing Image...");
                
                // Simulate backend analysis delay
                setTimeout(() => {
                    const fakeCount = Math.floor(Math.random() * 200) + 10;
                    setCrowdCount(fakeCount);
                    setDensityLevel(fakeCount > 150 ? "Critical" : (fakeCount > 50 ? "Crowded" : "Normal"));
                    setStatus("Analysis Complete");
                }, 2000);
            };
            reader.readAsDataURL(file);
        }
    };

    const toggleLiveFeed = () => {
        if (!isLive) {
            setImagePreview(null);
            setIsLive(true);
            setStatus("Connecting to Webcam...");
            // Simulated webcam connection
            setTimeout(() => {
                setStatus("Live Feed Active");
                setCrowdCount(Math.floor(Math.random() * 50) + 20); // initial sim
            }, 1000);
        } else {
            setIsLive(false);
            setStatus("Ready to Analyze");
            setCrowdCount(0);
            setDensityLevel("Normal");
        }
    };

    return (
        <div className="app-container">
            <header>
                <div className="logo-section">
                    <i className="ph ph-users-three logo-icon"></i>
                    <span className="logo-text">CrowdSense</span>
                </div>
                <div className="status-badge" style={{
                    borderColor: status === "Disconnected" ? "rgba(239, 68, 68, 0.3)" : 
                                 status.includes("Active") ? "rgba(16, 185, 129, 0.3)" : "rgba(245, 158, 11, 0.3)",
                    color: status === "Disconnected" ? "var(--danger)" : 
                           status.includes("Active") ? "var(--success)" : "var(--warning)",
                    background: status === "Disconnected" ? "rgba(239, 68, 68, 0.1)" : 
                                status.includes("Active") ? "rgba(16, 185, 129, 0.1)" : "rgba(245, 158, 11, 0.1)"
                }}>
                    <div className="status-dot" style={{
                        backgroundColor: status === "Disconnected" ? "var(--danger)" : 
                                         status.includes("Active") ? "var(--success)" : "var(--warning)",
                        boxShadow: `0 0 8px ${status === "Disconnected" ? "var(--danger)" : status.includes("Active") ? "var(--success)" : "var(--warning)"}`
                    }}></div>
                    {status}
                </div>
            </header>

            <main>
                <div className="glass-panel">
                    <div className="panel-title">
                        <i className="ph ph-video-camera"></i> Scene Monitor
                    </div>
                    
                    <div className="main-view-container">
                        {!imagePreview && !isLive ? (
                            <div className="placeholder-view">
                                <i className="ph ph-image placeholder-icon"></i>
                                <h3>No Feed Selected</h3>
                                <p style={{ color: "var(--text-secondary)" }}>Upload an image or start the webcam stream.</p>
                            </div>
                        ) : isLive ? (
                            <div className="placeholder-view">
                                <i className="ph ph-webcam placeholder-icon" style={{color: "var(--success)"}}></i>
                                <h3>Simulated Webcam Active</h3>
                                <p style={{ color: "var(--text-secondary)" }}>(Requires backend API connection for real feed)</p>
                            </div>
                        ) : (
                            <img src={imagePreview} alt="Analyzed scene" className="video-feed" style={{ filter: viewMode === 'heatmap' ? 'sepia(1) hue-rotate(-50deg) saturate(3) blur(2px)' : 'none' }} />
                        )}

                        {(imagePreview || isLive) && (
                            <div className="view-controls">
                                <button 
                                    className={`control-btn ${viewMode === 'feed' ? 'active' : ''}`}
                                    onClick={() => setViewMode('feed')}
                                >
                                    Raw Feed
                                </button>
                                <button 
                                    className={`control-btn ${viewMode === 'heatmap' ? 'active' : ''}`}
                                    onClick={() => setViewMode('heatmap')}
                                    title="Requires Backend API"
                                >
                                    Heatmap
                                </button>
                            </div>
                        )}
                    </div>
                </div>

                <div className="glass-panel metrics-container">
                    <div className="panel-title">
                        <i className="ph ph-chart-line-up"></i> Analytics
                    </div>

                    <div className={`metric-card ${getStatusClass(crowdCount)}`}>
                        <div className="metric-label">Estimated Count</div>
                        <div className="metric-value">{crowdCount}</div>
                    </div>

                    <div className="metric-card">
                        <div className="metric-label">Density Status</div>
                        <div className="metric-value" style={{ 
                            fontSize: '2rem',
                            color: densityLevel === 'Critical' ? 'var(--danger)' : 
                                   densityLevel === 'Crowded' ? 'var(--warning)' : 'var(--success)',
                            background: 'none',
                            WebkitTextFillColor: 'unset'
                        }}>
                            {densityLevel}
                        </div>
                    </div>

                     <div className="metric-card">
                        <div className="metric-label">Anomalies Detected</div>
                        <div className="metric-value" style={{ 
                            fontSize: '2rem', 
                            color: anomalyDetected ? 'var(--danger)' : 'var(--success)',
                            background: 'none',
                            WebkitTextFillColor: 'unset'
                        }}>
                            {anomalyDetected ? "YES" : "NO"}
                        </div>
                    </div>

                    <div className="controls-section">
                        <input 
                            type="file" 
                            accept="image/*" 
                            ref={fileInputRef}
                            onChange={handleFileChange}
                        />
                        <button className="action-btn" onClick={handleUploadClick}>
                            <i className="ph ph-upload-simple"></i> Upload Image
                        </button>
                        <button className={`action-btn ${isLive ? 'primary' : ''}`} onClick={toggleLiveFeed}>
                            <i className={isLive ? "ph ph-stop-circle" : "ph ph-video-camera"}></i> 
                            {isLive ? "Stop Stream" : "Start Live Webcam"}
                        </button>
                    </div>
                </div>
            </main>
        </div>
    );
}

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);
