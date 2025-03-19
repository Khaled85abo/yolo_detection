# single_camera/routes.py
from flask import jsonify, request, Response, render_template

def register_routes(server):
    """Register all HTTP routes for the application"""
    app = server.app  # Get the Flask app instance
    
    # Home/index route
    @app.route('/')
    def index():
        return render_template('index.html')
    
    # API route for getting plank status
    @app.route('/api/status', methods=['GET'])
    def get_status():
        return jsonify(server.plank_status.to_dict())
    
    # Video stream route
    @app.route('/video_feed/<camera_id>')
    def video_feed(camera_id):
        return Response(
            server.camera_manager.generate_frames(camera_id),
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )
    
    # WebSocket route
    @app.route('/ws')
    def websocket_route():
        return server.websocket_handler.handle_connection()
    
    # Rules configuration route
    @app.route('/api/rules', methods=['GET', 'POST'])
    def rules():
        if request.method == 'GET':
            return jsonify(server.rule_engine.get_rules())
        elif request.method == 'POST':
            data = request.json
            success = server.rule_engine.update_rules(data)
            return jsonify({"success": success})
    
    # Add more routes as needed...