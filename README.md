# persian_kasre_ezafe

Building image of the project from the given Dockerfile:

`sudo docker build -t ImageName:Tag .`

Run the the image after building to run the app.py:

`sudo docker run -it -p 5000:5000 ImageName:Tag`

Sending Request:

`curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"text": "من امشب زندگی رو خیلی زیبا و جنجالی میبینم. چون فوتبال بارسلونا قراره پخش بشه"}'
`

