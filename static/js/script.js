// static/js/script.js
document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const uploadForm = document.getElementById('upload-form');
    const imageUpload = document.getElementById('image-upload');
    const predictBtn = document.getElementById('predict-btn');
    const webcamToggle = document.getElementById('webcam-toggle');
    const webcamContainer = document.getElementById('webcam-container');
    const webcam = document.getElementById('webcam');
    const captureBtn = document.getElementById('capture-btn');
    const loadingDiv = document.getElementById('loading');
    const resultsContainer = document.getElementById('results-container');
    const predictionResults = document.getElementById('prediction-results');
    const predictionImage = document.getElementById('prediction-image');
    const predictionName = document.getElementById('prediction-name');
    const predictionConfidence = document.getElementById('prediction-confidence');
    const confidenceText = document.getElementById('confidence-text');
    const allProbabilities = document.getElementById('all-probabilities');
    
    // Webcam setup
    let stream = null;
    
    // Handle form submission
    uploadForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const file = imageUpload.files[0];
        if (!file) {
            alert('Please select an image file');
            return;
        }
        
        // Show loading
        showLoading();
        
        // Create form data
        const formData = new FormData();
        formData.append('file', file);
        
        // Send request
        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                throw new Error(data.error);
            }
            displayResults(data, file);
        })
        .catch(error => {
            hideLoading();
            alert('Error: ' + error.message);
        });
    });
    
    // Webcam toggle
    webcamToggle.addEventListener('click', function() {
        if (webcamContainer.classList.contains('d-none')) {
            startWebcam();
        } else {
            stopWebcam();
        }
    });
    
    // Capture button
    captureBtn.addEventListener('click', function() {
        captureImage();
    });
    
    // Start webcam
    function startWebcam() {
        navigator.mediaDevices.getUserMedia({ video: true, audio: false })
            .then(function(mediaStream) {
                stream = mediaStream;
                webcam.srcObject = mediaStream;
                webcamContainer.classList.remove('d-none');
                captureBtn.disabled = false;
                webcamToggle.innerHTML = '<i class="fas fa-times me-2"></i>Stop Webcam';
                webcamToggle.classList.replace('btn-outline-secondary', 'btn-outline-danger');
            })
            .catch(function(err) {
                console.error("Error accessing webcam: " + err);
                alert("Cannot access webcam. Make sure it's enabled and you've given permission.");
            });
    }
    
    // Stop webcam
    function stopWebcam() {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            stream = null;
        }
        webcam.srcObject = null;
        webcamContainer.classList.add('d-none');
        webcamToggle.innerHTML = '<i class="fas fa-camera me-2"></i>Use Webcam';
        webcamToggle.classList.replace('btn-outline-danger', 'btn-outline-secondary');
    }
    
    // Capture image from webcam
    function captureImage() {
        // Create canvas element
        const canvas = document.createElement('canvas');
        canvas.width = webcam.videoWidth;
        canvas.height = webcam.videoHeight;
        
        // Draw current video frame to canvas
        const ctx = canvas.getContext('2d');
        ctx.drawImage(webcam, 0, 0, canvas.width, canvas.height);
        
        // Convert to base64
        const imageData = canvas.toDataURL('image/jpeg');
        
        // Show loading
        showLoading();
        
        // Send request
        fetch('/webcam_predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ image: imageData })
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                throw new Error(data.error);
            }
            
            // Create blob from base64
            const byteString = atob(imageData.split(',')[1]);
            const ab = new ArrayBuffer(byteString.length);
            const ia = new Uint8Array(ab);
            for (let i = 0; i < byteString.length; i++) {
                ia[i] = byteString.charCodeAt(i);
            }
            const blob = new Blob([ab], { type: 'image/jpeg' });
            const webcamFile = new File([blob], "webcam-capture.jpg", { type: 'image/jpeg' });
            
            displayResults(data, webcamFile);
        })
        .catch(error => {
            hideLoading();
            alert('Error: ' + error.message);
        });
    }
    
    // Show loading state
    function showLoading() {
        loadingDiv.classList.remove('d-none');
        resultsContainer.classList.add('d-none');
        predictionResults.classList.add('d-none');
    }
    
    // Hide loading state
    function hideLoading() {
        loadingDiv.classList.add('d-none');
        resultsContainer.classList.remove('d-none');
    }
    
    // Display results
    function displayResults(data, file) {
        hideLoading();
        predictionResults.classList.remove('d-none');
        
        // Set image
        let imageUrl;
        if (data.image_path) {
            // If we have a server path
            imageUrl = data.image_path;
        } else {
            // If we have a file object (webcam)
            imageUrl = URL.createObjectURL(file);
        }
        predictionImage.src = imageUrl;
        
        // Set prediction details
        const prediction = data.prediction;
        predictionName.textContent = prediction.class;
        
        // Set confidence bar
        const confidence = prediction.confidence;
        predictionConfidence.style.width = confidence + '%';
        confidenceText.textContent = `Confidence: ${confidence.toFixed(2)}%`;
        
        // Adjust bar color based on confidence level
        if (confidence >= 80) {
            predictionConfidence.className = 'progress-bar bg-success';
        } else if (confidence >= 50) {
            predictionConfidence.className = 'progress-bar bg-warning';
        } else {
            predictionConfidence.className = 'progress-bar bg-danger';
        }
        
        // Display all probabilities
        allProbabilities.innerHTML = '';
        prediction.all_probabilities.sort((a, b) => b.probability - a.probability);
        
        prediction.all_probabilities.forEach(item => {
            const div = document.createElement('div');
            div.className = 'probability-item';
            div.innerHTML = `
                <div class="d-flex justify-content-between">
                    <small>${item.name}</small>
                    <small>${item.probability.toFixed(2)}%</small>
                </div>
                <div class="progress probability-bar">
                    <div class="progress-bar ${getColorClass(item.probability)}" 
                         style="width: ${item.probability}%"></div>
                </div>
            `;
            allProbabilities.appendChild(div);
        });
    }
    
    // Get color class based on probability
    function getColorClass(probability) {
        if (probability >= 80) return 'bg-success';
        if (probability >= 50) return 'bg-warning';
        if (probability >= 20) return 'bg-info';
        return 'bg-danger';
    }
});