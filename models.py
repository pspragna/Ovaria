# models.py
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin

# Single SQLAlchemy instance to import in app.py
db = SQLAlchemy()

# ------------------------------
# User Model
# ------------------------------
class User(UserMixin, db.Model):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)

    # Flask-Login requires get_id() method (UserMixin provides it)
    def __repr__(self):
        return f"<User {self.username}, admin={self.is_admin}>"

# ------------------------------
# TrackingEntry Model
# ------------------------------
class TrackingEntry(db.Model):
    __tablename__ = 'tracking_entries'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    date = db.Column(db.String(20), nullable=False)
    step_count = db.Column(db.Integer, default=0)
    sleep_hours = db.Column(db.Float, default=0.0)
    balanced_meal = db.Column(db.Integer, default=0)
    junk_food = db.Column(db.Integer, default=0)
    notes = db.Column(db.Text, default='')

    # Relationship back to User
    user = db.relationship('User', backref=db.backref('tracking_entries', lazy=True))

    def __repr__(self):
        return f"<TrackingEntry user_id={self.user_id} date={self.date}>"
