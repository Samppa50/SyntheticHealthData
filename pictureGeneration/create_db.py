from api import db, app, Picture


with app.app_context():
    db.create_all()