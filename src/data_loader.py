import pandas as pd

def load_airports():
    """Load OpenFlights airports.dat from official URL.
    Returns DataFrame with IATA, lat, lon, etc.
    Filters valid entries only."""
    url = "https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat"
    columns = [
        'id', 'name', 'city', 'country', 'iata', 'icao',
        'lat', 'lon', 'alt', 'timezone', 'dst', 'tz_db', 'type', 'source'
    ]
    df = pd.read_csv(
        url, header=None, names=columns, quotechar='"',
        na_values='\\N', encoding='utf-8'
    )
    # Clean: keep only airports with valid IATA and coordinates
    df = df.dropna(subset=['iata', 'lat', 'lon'])
    df['iata'] = df['iata'].astype(str).str.upper().str.strip()
    print(f"Loaded {len(df)} airports.")
    return df

def load_routes():
    """Load OpenFlights routes.dat. Keep only nonstop (stops==0) flights."""
    url = "https://raw.githubusercontent.com/jpatokal/openflights/master/data/routes.dat"
    columns = [
        'airline', 'airline_id', 'source', 'source_id',
        'dest', 'dest_id', 'codeshare', 'stops', 'equipment'
    ]
    df = pd.read_csv(
        url, header=None, names=columns, quotechar='"',
        na_values='\\N', encoding='utf-8'
    )
    df['source'] = df['source'].astype(str).str.upper().str.strip()
    df['dest'] = df['dest'].astype(str).str.upper().str.strip()
    df = df[df['stops'] == 0]  # Only direct flights
    print(f"Loaded {len(df)} direct routes.")
    return df