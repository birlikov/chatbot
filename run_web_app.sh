docker run -v "$PWD":/usr/workdir -p 80:80 -ti chatbot_docker_image --name chatbot /bin/bash -c "python web_app.py"