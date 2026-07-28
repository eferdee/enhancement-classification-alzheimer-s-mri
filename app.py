# app.py
"""
Flask application untuk deployment sistem klasifikasi Alzheimer's MRI.

Aplikasi ini menyediakan REST API untuk operasi enhancement dan classification
pada citra MRI menggunakan NAFNet dan ResNet18.
"""

import os
import base64
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

from inference_service import InferenceService

# FLASK APPLICATION CLASS
class FlaskApp:
    """
    Controller class yang mengelola Flask application lifecycle,
    routing, dan dependency terhadap InferenceService.

    Attributes:
        app: Flask application instance
        inference_service: InferenceService untuk operasi inferensi
    """

    def __init__(
        self,
        model_enhance_path='nafnet_best.pth',
        model_classifier_path='best_baseline_classifier.pth',
        device='cpu'
    ):
        """
        Inisialisasi FlaskApp.

        Args:
            model_enhance_path: Path ke NAFNet weights
            model_classifier_path: Path ke ResNet18 weights
            device: Device untuk inference ('cpu' atau 'cuda')
        """
        self.app = Flask(__name__)
        CORS(self.app)

        print("=" * 70)
        print("INITIALIZING FLASK APPLICATION")
        print("=" * 70)

        # Initialize InferenceService
        self.inference_service = InferenceService(
            model_enhance_path=model_enhance_path,
            model_classifier_path=model_classifier_path,
            device=device
        )
        self.inference_service.load_models()

        # Register routes
        self._register_routes()

    # ROUTE REGISTRATION
    def _register_routes(self):
        """Mendaftarkan semua route ke Flask app instance."""
        self.app.add_url_rule(
            '/',
            'index',
            self.index
        )
        self.app.add_url_rule(
            '/predict',
            'predict',
            self.predict,
            methods=['POST']
        )
        self.app.add_url_rule(
            '/enhance',
            'enhance',
            self.enhance,
            methods=['POST']
        )

    # ROUTES
    def index(self):
        """
        Serve main HTML interface.

        Returns:
            Rendered HTML template
        """
        return render_template('index.html')

    def predict(self):
        """
        Endpoint untuk klasifikasi citra MRI mentah (baseline).

        Expected request:
            - Method: POST
            - Content-Type: multipart/form-data
            - Body: file (image file)

        Returns:
            JSON: {prediction, confidence, probabilities} atau error
        """
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        result = self.inference_service.classify_image(file)

        if result:
            return jsonify(result), 200
        else:
            return jsonify({'error': 'Gagal melakukan klasifikasi (Cek log server).'}), 500

    def enhance(self):
        """
        Endpoint untuk enhancement citra MRI menggunakan NAFNet.

        Expected request:
            - Method: POST
            - Content-Type: multipart/form-data
            - Body: file (image file)

        Returns:
            JSON: {success, message, enhanced_image_base64} atau error
        """
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        enhanced_bytes = self.inference_service.enhance_image(file)

        if enhanced_bytes:
            base64_encoded_image = base64.b64encode(enhanced_bytes).decode('utf-8')
            return jsonify({
                'success': True,
                'message': 'Gambar berhasil ditingkatkan kualitasnya.',
                'enhanced_image_base64': base64_encoded_image
            }), 200
        else:
            return jsonify({'error': 'Gagal memproses peningkatan gambar (Cek terminal untuk detail)'}), 500

    # RUN
    def run(self, debug=False, host='0.0.0.0', port=5000):
        """
        Menjalankan Flask development server.

        Args:
            debug: Debug mode
            host: Host address
            port: Port number
        """
        print("\n" + "=" * 70)
        print("FLASK APPLICATION READY")
        print(f"Access the interface at: http://127.0.0.1:{port}")
        print("=" * 70 + "\n")
        self.app.run(debug=debug, host=host, port=port)

# MAIN
if __name__ == '__main__':
    flask_app = FlaskApp(
        model_enhance_path='nafnet_best.pth',
        model_classifier_path='best_baseline_classifier.pth',
        device='cpu'
    )
    flask_app.run(debug=False, host='0.0.0.0', port=5000)