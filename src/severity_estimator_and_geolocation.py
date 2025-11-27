
import re
import pandas as pd
import folium
from geopy.geocoders import Nominatim
import time


def estimate_severity(text):
    """Estimate severity from text"""
    text_lower = text.lower()

    high_kws = ['catastrophic', 'devastating', 'massive', 'destroyed', 'deaths',
                'casualties', 'collapsed', 'critical', 'emergency', 'urgent',
                'evacuation', 'fatalities', 'killed', 'trapped', 'thousands']

    medium_kws = ['damage', 'injured', 'warning', 'alert', 'spreading', 'affected']

    low_kws = ['minor', 'small', 'slight', 'contained', 'under control', 'safe']

    high_count = sum(1 for kw in high_kws if kw in text_lower)
    low_count = sum(1 for kw in low_kws if kw in text_lower)

    if high_count >= 2:
        return 'HIGH', min(0.9, 0.6 + high_count * 0.1)
    elif low_count >= 1 and high_count == 0:
        return 'LOW', 0.7
    else:
        return 'MEDIUM', 0.65

print("✅ Severity estimator ready!")

print("\n📍 Setting up geolocation...")

geolocator = Nominatim(user_agent="disasterscope_ai_v2")

KNOWN_LOCATIONS = {
    'california': {'lat': 36.7783, 'lng': -119.4179, 'name': 'California, USA'},
    'los angeles': {'lat': 34.0522, 'lng': -118.2437, 'name': 'Los Angeles, USA'},
    'san francisco': {'lat': 37.7749, 'lng': -122.4194, 'name': 'San Francisco, USA'},
    'new york': {'lat': 40.7128, 'lng': -74.0060, 'name': 'New York, USA'},
    'texas': {'lat': 31.9686, 'lng': -99.9018, 'name': 'Texas, USA'},
    'houston': {'lat': 29.7604, 'lng': -95.3698, 'name': 'Houston, USA'},
    'florida': {'lat': 27.6648, 'lng': -81.5158, 'name': 'Florida, USA'},
    'miami': {'lat': 25.7617, 'lng': -80.1918, 'name': 'Miami, USA'},
    'japan': {'lat': 36.2048, 'lng': 138.2529, 'name': 'Japan'},
    'tokyo': {'lat': 35.6762, 'lng': 139.6503, 'name': 'Tokyo, Japan'},
    'india': {'lat': 20.5937, 'lng': 78.9629, 'name': 'India'},
    'delhi': {'lat': 28.6139, 'lng': 77.2090, 'name': 'Delhi, India'},
    'mumbai': {'lat': 19.0760, 'lng': 72.8777, 'name': 'Mumbai, India'},
    'philippines': {'lat': 12.8797, 'lng': 121.7740, 'name': 'Philippines'},
    'australia': {'lat': -25.2744, 'lng': 133.7751, 'name': 'Australia'},
    'indonesia': {'lat': -0.7893, 'lng': 113.9213, 'name': 'Indonesia'},
    'china': {'lat': 35.8617, 'lng': 104.1954, 'name': 'China'},
    'turkey': {'lat': 38.9637, 'lng': 35.2433, 'name': 'Turkey'},
    'nepal': {'lat': 28.3949, 'lng': 84.1240, 'name': 'Nepal'},
    'haiti': {'lat': 18.9712, 'lng': -72.2852, 'name': 'Haiti'},
    'chile': {'lat': -35.6751, 'lng': -71.5430, 'name': 'Chile'},
}

def extract_location(text):
    """Extract location from text with geocoding"""
    text_lower = text.lower()

    # Check known locations first (fast)
    for loc_key, loc_data in KNOWN_LOCATIONS.items():
        if loc_key in text_lower:
            return {
                'found': True,
                'name': loc_data['name'],
                'lat': loc_data['lat'],
                'lng': loc_data['lng'],
                'confidence': 0.9
            }

   
    patterns = [
        r'(?:in|at|near|hits|strikes|devastates)\s+([A-Z][a-zA-Z\s]+?)(?:[,.\!\?]|$)',
        r'([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)\s+(?:earthquake|flood|fire|hurricane)',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            location_name = matches[0].strip()
            try:
                location = geolocator.geocode(location_name, timeout=5)
                if location:
                    return {
                        'found': True,
                        'name': location.address.split(',')[0],
                        'lat': location.latitude,
                        'lng': location.longitude,
                        'confidence': 0.75
                    }
            except:
                pass

    return {'found': False, 'name': None, 'lat': None, 'lng': None, 'confidence': 0}

print("✅ Geolocation ready!")


def create_disaster_map(location, disaster_type, severity, save_path='outputs/disaster_map.html'):
    """
    Create interactive map with:
    - RED zone at epicenter (critical)
    - ORANGE zone around it (warning)
    - YELLOW zone outer (caution)
    """
    if not location['found']:
        print("⚠️ No location found - cannot create map")
        return None

    lat, lng = location['lat'], location['lng']

    
    m = folium.Map(location=[lat, lng], zoom_start=10)

    
    zone_sizes = {
        'HIGH': {'red': 5000, 'orange': 15000, 'yellow': 30000},
        'MEDIUM': {'red': 3000, 'orange': 10000, 'yellow': 20000},
        'LOW': {'red': 1000, 'orange': 5000, 'yellow': 10000},
    }
    sizes = zone_sizes.get(severity, zone_sizes['MEDIUM'])

    
    folium.Circle(
        location=[lat, lng],
        radius=sizes['yellow'],
        color='#FFD700',
        fill=True,
        fill_color='yellow',
        fill_opacity=0.2,
        popup='⚠️ Caution Zone'
    ).add_to(m)

    
    folium.Circle(
        location=[lat, lng],
        radius=sizes['orange'],
        color='orange',
        fill=True,
        fill_color='orange',
        fill_opacity=0.3,
        popup='🟠 Warning Zone'
    ).add_to(m)

    
    folium.Circle(
        location=[lat, lng],
        radius=sizes['red'],
        color='red',
        fill=True,
        fill_color='red',
        fill_opacity=0.4,
        popup='🔴 Critical Zone - Immediate Danger'
    ).add_to(m)

    
    popup_html = f"""
    <div style="width: 200px; font-family: Arial;">
        <h4 style="color: red; margin: 0;">🚨 {disaster_type.upper()}</h4>
        <hr style="margin: 5px 0;">
        <p><b>📍 Location:</b> {location['name']}</p>
        <p><b>⚠️ Severity:</b> <span style="color: {'red' if severity=='HIGH' else 'orange' if severity=='MEDIUM' else 'green'};">{severity}</span></p>
        <p><b>📐 Coordinates:</b><br>{lat:.4f}, {lng:.4f}</p>
    </div>
    """

    folium.Marker(
        location=[lat, lng],
        popup=folium.Popup(popup_html, max_width=250),
        icon=folium.Icon(color='red', icon='exclamation-triangle', prefix='fa'),
        tooltip=f"🚨 {disaster_type.upper()} - Click for details"
    ).add_to(m)

   
    title_html = f'''
    <div style="position: fixed; top: 10px; left: 50px;
                background: white; padding: 10px 15px;
                border: 3px solid red; border-radius: 10px;
                z-index: 9999; font-family: Arial;">
        <h3 style="margin: 0; color: red;">🚨 DisasterScope AI Alert</h3>
        <p style="margin: 5px 0 0 0;"><b>{disaster_type.upper()}</b> detected in <b>{location['name']}</b></p>
        <p style="margin: 0; color: {'red' if severity=='HIGH' else 'orange'};">Severity: {severity}</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))

    legend_html = '''
    <div style="position: fixed; bottom: 30px; right: 30px;
                background: white; padding: 10px;
                border: 2px solid gray; border-radius: 8px;
                z-index: 9999; font-family: Arial; font-size: 12px;">
        <b>Legend</b><br>
        🔴 Critical Zone<br>
        🟠 Warning Zone<br>
        🟡 Caution Zone
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

   
    import os
    os.makedirs('outputs', exist_ok=True)
    m.save(save_path)
    print(f"✅ Map saved to: {save_path}")

    return m

print("✅ Map generator ready!")

def generate_alert(disaster_type, severity, location_name):
    """Generate alert message"""

    alerts = {
        'HIGH': {
            'emoji': '🚨',
            'level': 'CRITICAL',
            'message': f"CRITICAL: {disaster_type.upper()} in {location_name}! Immediate action required!",
            'action': 'Evacuate immediately. Follow emergency services instructions.'
        },
        'MEDIUM': {
            'emoji': '⚠️',
            'level': 'WARNING',
            'message': f"WARNING: {disaster_type.upper()} reported in {location_name}. Stay alert!",
            'action': 'Prepare emergency supplies. Monitor official channels.'
        },
        'LOW': {
            'emoji': 'ℹ️',
            'level': 'ADVISORY',
            'message': f"ADVISORY: Minor {disaster_type} activity in {location_name}.",
            'action': 'Stay informed. No immediate action needed.'
        }
    }

    return alerts.get(severity, alerts['MEDIUM'])

print("✅ Alert generator ready!")


def analyze_text(tweet_text):
    """Complete text analysis pipeline"""

    cleaned = clean_tweet(tweet_text)

   
    disaster_result = disaster_classifier.classify(cleaned)

   
    location = extract_location(tweet_text)

   
    severity, severity_conf = estimate_severity(cleaned)

    
    loc_name = location['name'] if location['found'] else 'Unknown Location'
    alert = generate_alert(disaster_result['disaster_type'], severity, loc_name)

    return {
        'original_text': tweet_text,
        'cleaned_text': cleaned,
        'disaster_type': disaster_result['disaster_type'],
        'disaster_confidence': disaster_result['confidence'],
        'location': location,
        'severity': severity,
        'severity_confidence': severity_conf,
        'alert': alert
    }

print("\n Complete text analysis ready!")



test_tweets = [
    "BREAKING: Massive earthquake hits Tokyo, Japan! Buildings collapsed, thousands trapped!",
    "Devastating wildfire spreading near Los Angeles. 50,000 evacuated!",
    "Flash flood warning for Houston, Texas. Roads completely submerged!",
    "Hurricane makes landfall in Florida with Category 4 winds!",
    "Minor tremors felt in California. No damage reported.",
]

for i, tweet in enumerate(test_tweets, 1):
    result = analyze_text(tweet)

    print(f"\n{'='*50}")
    print(f"📝 Tweet {i}: {tweet[:50]}...")
    print(f"   🌪️ Disaster: {result['disaster_type'].upper()} ({result['disaster_confidence']*100:.0f}%)")
    print(f"   📍 Location: {result['location']['name'] if result['location']['found'] else 'Not found'}")
    print(f"   ⚠️ Severity: {result['severity']}")
    print(f"   {result['alert']['emoji']} {result['alert']['level']}: {result['alert']['message'][:50]}...")




result = analyze_text(test_tweets[0])
if result['location']['found']:
    disaster_map = create_disaster_map(
        result['location'],
        result['disaster_type'],
        result['severity']
    )
   

  
    from IPython.display import display, HTML
    display(disaster_map)



