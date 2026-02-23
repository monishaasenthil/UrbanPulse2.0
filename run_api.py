"""
Quick script to run the Urban Pulse API server
"""
from api.app import run_server

if __name__ == '__main__':
    run_server(host='0.0.0.0', port=5000, debug=True)
