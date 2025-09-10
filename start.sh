#!/bin/bash
cd ml-diagnosis-tool-main
exec gunicorn --bind 0.0.0.0:$PORT app:app
