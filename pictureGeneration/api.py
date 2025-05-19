from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_restful import Api, Resource,reqparse, fields, marshal_with, abort

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db = SQLAlchemy(app)

class Picture(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)

    def __repr__(self):
        return f'<Picture {self.name}>'

user_args = reqparse.RequestParser()
user_args.add_argument("name", type=str, required=True , help="Name of the picture cannot be blank")

userFields = {
    'id': fields.Integer,
    'name': fields.String
}

class PictureResource(Resource):
    @marshal_with(userFields)
    def get(self, id):
        picture = Picture.query.get(id)
        if not picture:
            abort(404, message="Picture not found")
        return {"id": picture.id, "name": picture.name}
    @marshal_with(userFields)
    def post(self):
        args = user_args.parse_args()
        new_picture = Picture(name=args["name"])
        db.session.add(new_picture)
        db.session.commit()
        return new_picture, 201  # Return only the new picture
    @marshal_with(userFields)
    def delete(self, id):
        picture = Picture.query.get(id)
        if not picture:
            abort(404, message="Picture not found")
        db.session.delete(picture)
        db.session.commit()
        return '', 204
api = Api(app)
api.add_resource(PictureResource, "/picture/<int:id>", "/picture")



@app.route("/")
def hello_world():
    return "<p>This will be our picture api</p>"

app.run(debug=True ,host="0.0.0.0", port=5002)