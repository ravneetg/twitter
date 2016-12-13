from flask import Flask, request
from flask_restful import Resource, Api
from sqlalchemy import create_engine
from json import dumps
import pyhs2

#Create a engine for connecting to SQLite3.
#Assuming salaries.db is in your app root folder

#e = create_engine('sqlite:///salaries.db')

app = Flask(__name__)
api = Api(app)

class Movie_Meta(Resource):
    def get(self):
        # with pyhs2.connect(host='192.168.1.221',
        # port=10000,
        # authMechanism="PLAIN",
        # user='w205user',
        # password='',
        # database='default') as conn:
        #     print "Hi"
        #     with conn.cursor() as cur:
        #         #Show databases
        #         print cur.getDatabases()
        #
        #         #Execute query
        #         transp.execute("select * from movie_hashtags")
        #
        #         #Return column info from query
        #         print cur.getSchema()
        #
        #         #Fetch table results
        #         for i in cur.fetch():
        #             print i
        #         #Connect to databse
        #         #conn = e.connect()
        #         #Perform query and return JSON data
        #         #query = conn.execute("select distinct DEPARTMENT from salaries")
        #         #replace below line with sql query from DB
        #print "Hi2"
        lines = [line.rstrip('\n') for line in open('movienames.txt')]
        return {'moviename': [lines]}

api.add_resource(Movie_Meta, '/movies')

if __name__ == '__main__':
            #print "Hi3"
    app.run()
