This is a cache used to store weather data downloaded from open meteorology
The API is slow, so .parquet files are downloaded once and used from this directory after that

Since the data is historical, there is no need to refresh these.
Refreshing them will be difficult - any request duration over 2 weeks technically requires seperate API request
If you run late at night the chance of being rate limited is much lower

Again - data is complete so no real need to ever rerun.