"""Module containing abstractions for interacting with whatever datastore is
used to store bird records and metrics. Could call this "gullet" or something to
stick with the bird theme lol.
"""

class ZoneEvent:
    pass

class ZoneVisit:
    pass

class Gullet:
    """Database wrapper. Sits on top of database ORM or some other storage
    abstraction.
    """

    def create_zone_event(zone_event: ZoneEvent):
        # Return some unique ID for the zone event? Like some kind of primary
        # key?
        pass
    
    def create_zone_visit(zone_visit: ZoneVisit):
        pass

# For the "database wrapper," I just want this to be an abstraction over the
# database--but I don't want it to be the ONLY WAY that you're able to or
# allowed to interact with the database. Like, if you have an SQL database, I
# still think you should be able to write code to query it for computing
# metrics, for example. However, even then, that code you write to "query" your
# database of choice *should* be database agnostic (via an ORM like SQLAlchemy).

# However, my goal for this wrapper isn't that--that's the job of an ORM. I want
# this to be a layer ABOVE an ORM for very common operations.

