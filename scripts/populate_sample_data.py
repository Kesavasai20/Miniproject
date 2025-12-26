"""
Script to populate the database with sample ARGO float data
"""
import sys
sys.path.insert(0, '.')

from database.postgres import get_db_manager
from database.models import Float, Profile, Measurement, FloatStatus
from datetime import datetime, timedelta
import random

def create_sample_data():
    """Create sample ARGO float data in the database"""
    db = get_db_manager()
    print('Creating sample floats in database...')

    with db.get_session() as session:
        # Sample floats matching Indian Ocean data
        sample_floats = [
            {'wmo_id': '2901337', 'lat': 15.5, 'lon': 68.3, 'status': 'active', 'institution': 'INCOIS', 'cycles': 150},
            {'wmo_id': '2901338', 'lat': 12.8, 'lon': 85.2, 'status': 'active', 'institution': 'INCOIS', 'cycles': 120},
            {'wmo_id': '2901339', 'lat': -5.2, 'lon': 70.1, 'status': 'inactive', 'institution': 'INCOIS', 'cycles': 80},
            {'wmo_id': '2901340', 'lat': 8.7, 'lon': 76.5, 'status': 'active', 'institution': 'INCOIS', 'cycles': 200},
            {'wmo_id': '2901341', 'lat': 20.1, 'lon': 65.3, 'status': 'active', 'institution': 'INCOIS', 'cycles': 95},
            {'wmo_id': '2901342', 'lat': -10.5, 'lon': 95.2, 'status': 'active', 'institution': 'IFREMER', 'cycles': 180},
            {'wmo_id': '2901343', 'lat': 5.3, 'lon': 55.8, 'status': 'lost', 'institution': 'IFREMER', 'cycles': 45},
        ]
        
        floats_created = 0
        profiles_created = 0
        measurements_created = 0
        
        for sf in sample_floats:
            # Check if float exists
            existing = session.query(Float).filter_by(wmo_id=sf['wmo_id']).first()
            if existing:
                print(f"Float {sf['wmo_id']} already exists, skipping...")
                continue
            
            # Create float
            status_map = {'active': FloatStatus.ACTIVE, 'inactive': FloatStatus.INACTIVE, 'lost': FloatStatus.LOST}
            float_obj = Float(
                wmo_id=sf['wmo_id'],
                current_latitude=sf['lat'],
                current_longitude=sf['lon'],
                deploy_latitude=sf['lat'] - random.uniform(0.5, 2.0),
                deploy_longitude=sf['lon'] - random.uniform(0.5, 2.0),
                deploy_date=datetime.now() - timedelta(days=random.randint(365, 1000)),
                last_position_date=datetime.now() - timedelta(days=random.randint(1, 30)),
                status=status_map.get(sf['status'], FloatStatus.ACTIVE),
                institution=sf['institution'],
                country='India' if sf['institution'] == 'INCOIS' else 'France',
                total_cycles=sf['cycles'],
                has_oxygen=random.choice([True, False]),
                has_chlorophyll=random.choice([True, False])
            )
            session.add(float_obj)
            session.flush()  # Get ID
            floats_created += 1
            print(f"Created float {sf['wmo_id']}")
            
            # Create 10 profiles per float
            for cycle in range(1, 11):
                profile = Profile(
                    float_id=float_obj.id,
                    cycle_number=cycle,
                    latitude=sf['lat'] + random.uniform(-0.5, 0.5),
                    longitude=sf['lon'] + random.uniform(-0.5, 0.5),
                    date_time=datetime.now() - timedelta(days=30 - cycle*3),
                    n_levels=100,
                    max_pressure=2000,
                    surface_temp=28.0 + random.uniform(-2, 2),
                    surface_salinity=35.0 + random.uniform(-0.5, 0.5)
                )
                session.add(profile)
                session.flush()
                profiles_created += 1
                
                # Create measurements for each profile
                for pressure in range(0, 2000, 20):
                    depth = pressure * 0.99
                    temp = 28 - 0.01 * pressure + random.uniform(-0.5, 0.5)
                    sal = 35 + 0.0015 * pressure + random.uniform(-0.1, 0.1)
                    oxy = max(50, 220 - 0.08 * pressure + random.uniform(-10, 10)) if pressure < 1500 else None
                    
                    meas = Measurement(
                        profile_id=profile.id,
                        pressure=float(pressure),
                        depth=float(depth),
                        temperature=float(temp),
                        salinity=float(sal),
                        oxygen=float(oxy) if oxy else None,
                        temp_qc=1,
                        sal_qc=1
                    )
                    session.add(meas)
                    measurements_created += 1
        
        session.commit()
        print(f"\n=== Summary ===")
        print(f"Floats created: {floats_created}")
        print(f"Profiles created: {profiles_created}")
        print(f"Measurements created: {measurements_created}")
        return floats_created, profiles_created, measurements_created


if __name__ == "__main__":
    create_sample_data()
