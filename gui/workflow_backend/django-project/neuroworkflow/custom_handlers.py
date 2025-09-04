from tornado import web
import json


class CORSHandler(web.RequestHandler):
    """CORS preflight handler for iframe embedding"""
    
    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin", "http://localhost:5173")
        self.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.set_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
        self.set_header("Access-Control-Allow-Credentials", "true")
    
    def options(self, *args):
        """Handle CORS preflight requests"""
        self.set_status(204)
        self.finish()


class AuthStatusHandler(CORSHandler):
    """Check authentication status for iframe embedding"""
    
    async def get(self):
        """Return authentication status"""
        user = self.current_user
        if user:
            self.write({
                'authenticated': True,
                'username': user.name,
                'server_url': f"/user/{user.name}/lab"
            })
        else:
            self.write({'authenticated': False})