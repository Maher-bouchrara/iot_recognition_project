# models/intruder.py

class Intruder:
    def __init__(self, name, date, location):
        self.name = name
        self.date = date
        self.location = location

    def to_dict(self):
        return {
            "name": self.name,
            "date": self.date,
            "location": self.location
        }
