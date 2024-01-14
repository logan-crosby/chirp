"""Functions for computing/extracting metrics from chirp feeder session data."""

# Should create an interface here that allows us to retrieve essentially
# identical records as were inserted to whatever database or datastorage medium
# is being used. That is, the records can be stored in whatever format you want,
# but what goes in and what comes out should have the same class structure.

# Therefore, we should define a class that specifies what those records look
# like--and then each database or datastore interface that we have can take that
# common datatype and store it however it needs to.

# Need that common datatype first. THEN, can take that datatype and compute
# metrics with it here.

# These functions shouldn't ever talk to the database--that's out of their
# scope. Instead, each metric should just expect a record or list of records
# from the database and compute a metric from them, or something to that effect.

# OR, these metrics should have some common interface that they can use to query
# the underlying datastore, if providing them records isn't flexible enough (it
# very well may not be). That interface they would hypothetically use to
# interact with the databsae would therefore return that common record datatype
# that they understand how to work with.