#!/bin/bash
#    pip install gunicorn
gunicorn --workers 1 --threads 4 --bind 0.0.0.0:8000 pi_stream_server:app
