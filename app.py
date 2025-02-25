from flask import Flask, jsonify, request
import os

# Initialize Flask app
app = Flask(__name__)

# Get configuration from environment variables
app.config['DEBUG'] = True

@app.route('/')
def home():
    return jsonify({
        "message": "Welcome to Flask API",
        "status": "success"
    })

@app.route('/health')
def health_check():
    return jsonify({
        "status": "healthy"
    })

@app.route('/generate-image', methods=['POST'])
def generate_image():
    # Get parameters from request, set defaults if not provided
    data = request.get_json()
    
    width = data.get('width', 1024)
    height = data.get('height', 1024)
    seed = data.get('seed', 0)
    steps = data.get('steps', 20)
    sampler_name = data.get('sampler_name', 'euler')
    scheduler = data.get('scheduler', 'simple')
    positive_prompt = data.get('positive_prompt', '')

    # For now, return a mock response
    # In a real implementation, you would integrate with an image generation service
    return jsonify({
        "status": "success",
        "data": {
            "image_url": "https://example.com/generated-image.jpg",
            "parameters": {
                "width": width,
                "height": height,
                "seed": seed,
                "steps": steps,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "positive_prompt": positive_prompt
            }
        }
    })

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port) 