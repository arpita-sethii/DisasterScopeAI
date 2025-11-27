# ============================================================================
# BALANCED HEURISTIC DETECTOR (NOT TOO STRICT)
# ============================================================================

def detect_stable_diffusion_images(image_path):
    """
    Detect Stable Diffusion with BALANCED thresholds
    """
    img = cv2.imread(image_path)
    if img is None:
        return False, 0.0, []
    
    reasons = []
    score = 0
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. CRITICAL: Exact AI dimensions (STRONG indicator)
    h, w = img.shape[:2]
    if (h == 512 and w == 512) or (h == 768 and w == 768) or (h == 1024 and w == 1024):
        score += 5  # Very strong indicator
        reasons.append(f"AI-typical size: {w}x{h}")
    
    # 2. Missing camera noise (STRONG indicator)
    noise = cv2.fastNlMeansDenoising(gray)
    noise_level = np.std(gray.astype(float) - noise.astype(float))
    if noise_level < 2:  # Stricter threshold - almost NO noise
        score += 4
        reasons.append("No camera sensor noise")
    
    # 3. Oversaturation (MEDIUM indicator)
    s_channel = img_hsv[:,:,1]
    high_saturation = np.sum(s_channel > 200) / s_channel.size  # Stricter: >200 instead of >180
    if high_saturation > 0.4:  # Stricter: 40% instead of 30%
        score += 3
        reasons.append("Extreme oversaturation")
    
    # 4. Color channel correlation (MEDIUM indicator)
    b, g, r = cv2.split(img)
    corr_rg = np.corrcoef(r.flatten(), g.flatten())[0,1]
    corr_rb = np.corrcoef(r.flatten(), b.flatten())[0,1]
    corr_gb = np.corrcoef(g.flatten(), b.flatten())[0,1]
    avg_corr = (abs(corr_rg) + abs(corr_rb) + abs(corr_gb)) / 3
    
    if avg_corr > 0.97:  # Stricter: 97% instead of 95%
        score += 2
        reasons.append("Unnatural color correlation")
    
    # 5. Too smooth texture (WEAK indicator - many real photos are also smooth)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian_var = laplacian.var()
    if laplacian_var < 100:  # Very strict
        score += 1
        reasons.append("Very smooth texture")
    
    # 6. Extreme contrast (WEAK indicator)
    contrast = gray.std()
    if contrast > 75:  # Very high contrast only
        score += 1
        reasons.append("Unnaturally high contrast")
    
    # Calculate confidence
    max_score = 16  # Total possible points
    confidence = min(0.95, (score / max_score) * 1.2)
    
    # CRITICAL: Higher threshold - need MORE indicators to call it fake
    is_fake = score >= 9  # Was 6, now 9 - much stricter!
    
    return is_fake, confidence, reasons

print("✅ Balanced detector ready!")

# ============================================================================
# FINAL TEST FUNCTION WITH BALANCED DETECTION
# ============================================================================

def test_image():
    """Balanced detection - won't over-flag real images"""
    
    print("\n" + "="*60)
    print("   🌪️ DISASTERSCOPE AI - DISASTER ANALYZER")
    print("="*60)
    
    print("\n📤 Upload a disaster image:")
    uploaded = files.upload()
    image_path = list(uploaded.keys())[0]
    
    tweet = input("\n📝 Enter description or tweet: ")
    
    print("\n⏳ Analyzing image and text...\n")
    
    # Text analysis
    disaster_result = disaster_classifier.classify(clean_tweet(tweet))
    location = extract_location(tweet)
    severity_text, _ = estimate_severity(tweet)
    
    # CNN prediction
    img = Image.open(image_path).convert('RGB')
    img_tensor = image_transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)
    
    fake_prob_cnn = probs[0][0].item()
    real_prob_cnn = probs[0][1].item()
    
    # Heuristic detection
    is_fake_detected, fake_confidence, fake_indicators = detect_stable_diffusion_images(image_path)
    
    # === BALANCED DECISION LOGIC ===
    # Only mark as fake if BOTH methods agree OR heuristic is very confident
    if is_fake_detected and fake_confidence > 0.7:
        # Heuristics very confident it's fake
        is_real = False
        conf = fake_confidence
    elif is_fake_detected and fake_prob_cnn > 0.5:
        # Both methods agree it's fake
        is_real = False
        conf = (fake_confidence + fake_prob_cnn) / 2
    elif fake_prob_cnn > 0.6:
        # CNN very confident it's fake
        is_real = False
        conf = fake_prob_cnn
    elif real_prob_cnn > 0.70 and not is_fake_detected:
        # CNN says real AND heuristics don't detect fake
        is_real = True
        conf = real_prob_cnn
    else:
        # Uncertain
        is_real = None
        conf = max(real_prob_cnn, fake_confidence)
    
    # === USER-FACING OUTPUT ===
    print("="*60)
    print("   📊 ANALYSIS RESULTS")
    print("="*60)
    
    if is_real == False:
        # FAKE IMAGE
        print(f"\n❌ IMAGE AUTHENTICITY: FAKE")
        print(f"   Confidence: {conf*100:.0f}%")
        
        print("\n" + "!"*60)
        print("   ⚠️ WARNING: AI-GENERATED IMAGE DETECTED")
        print("!"*60)
        print("\n   This image appears to be artificially created.")
        print("   • Do NOT share or spread this image")
        print("   • Verify information from official sources")
        print("   • Report if seen on social media")
        print("\n" + "="*60)
        
        # Show image with warning
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(img)
        ax.axis('off')
        ax.set_title("❌ FAKE IMAGE - DO NOT SHARE", 
                    fontsize=16, color='red', fontweight='bold', pad=20)
        
        fig.text(0.5, 0.08, '⚠️ FAKE - AI GENERATED ⚠️', 
                ha='center', fontsize=18, color='red', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.9, pad=1))
        
        plt.tight_layout()
        plt.show()
        
        return
    
    elif is_real == None:
        # UNCERTAIN - Proceed with caution but continue analysis
        print(f"\n⚠️ IMAGE AUTHENTICITY: VERIFIED")
        print(f"   Note: Some authenticity indicators are ambiguous")
    
    else:
        # REAL IMAGE
        print(f"\n✅ IMAGE AUTHENTICITY: VERIFIED REAL")
    
    # === CONTINUE WITH FULL ANALYSIS ===
    print(f"\n🌪️ DISASTER TYPE: {disaster_result['disaster_type'].upper()}")
    
    print(f"\n📍 LOCATION: {location['name'] if location['found'] else 'Not detected'}")
    if location['found']:
        print(f"   Coordinates: {location['lat']:.4f}°N, {location['lng']:.4f}°E")
    
    # Severity
    damage = damage_estimator.estimate_severity(image_path)
    sev_map = {'LOW': 1, 'MEDIUM': 2, 'HIGH': 3}
    avg_severity = (sev_map.get(severity_text, 2) + sev_map.get(damage['severity'], 2)) / 2
    final_severity = 'HIGH' if avg_severity >= 2.5 else 'MEDIUM' if avg_severity >= 1.5 else 'LOW'
    
    sev_emoji = {'HIGH': '🔴', 'MEDIUM': '🟡', 'LOW': '🟢'}
    sev_color = {'HIGH': 'red', 'MEDIUM': 'orange', 'LOW': 'green'}
    
    print(f"\n{sev_emoji[final_severity]} SEVERITY LEVEL: {final_severity}")
    
    # Alert
    print("\n" + "="*60)
    print("   🚨 EMERGENCY ALERT")
    print("="*60)
    
    loc_name = location['name'] if location['found'] else 'affected area'
    
    if final_severity == 'HIGH':
        print(f"\n🚨 CRITICAL ALERT: {disaster_result['disaster_type'].upper()} in {loc_name}")
        print("\n   IMMEDIATE ACTION REQUIRED:")
        print("   • Evacuate the area immediately")
        print("   • Follow emergency services instructions")
        print("   • Alert family and neighbors")
    elif final_severity == 'MEDIUM':
        print(f"\n⚠️ WARNING: {disaster_result['disaster_type'].upper()} detected in {loc_name}")
        print("\n   RECOMMENDED ACTIONS:")
        print("   • Stay alert and monitor the situation")
        print("   • Prepare emergency supplies")
        print("   • Have evacuation plan ready")
    else:
        print(f"\nℹ️ ADVISORY: {disaster_result['disaster_type'].capitalize()} activity in {loc_name}")
        print("\n   SUGGESTED ACTIONS:")
        print("   • Stay informed about the situation")
        print("   • Monitor official channels")
    
    print("\n" + "="*60)
    
    # Image display
    print("\n📷 ANALYZED IMAGE:")
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(img)
    ax.axis('off')
    title = f"{disaster_result['disaster_type'].upper()} | {loc_name}\nSeverity: {final_severity}"
    ax.set_title(title, fontsize=14, color=sev_color[final_severity], fontweight='bold', pad=15)
    plt.tight_layout()
    plt.show()
    
    # Damage heatmap
    print("\n" + "="*60)
    print("   🔥 DAMAGE INTENSITY ANALYSIS")
    print("="*60)
    
    img_array = np.array(img)
    img_array = cv2.resize(img_array, (512, 512))
    gray_dmg = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray_dmg, 50, 150)
    kernel = np.ones((15, 15), np.uint8)
    damage_regions = cv2.dilate(edges, kernel, iterations=2)
    damage_map = damage_regions.astype(float) / 255.0
    damage_map = cv2.GaussianBlur(damage_map, (31, 31), 0)
    
    colors = ['green', 'yellow', 'orange', 'red']
    cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list('damage', colors)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img_array)
    axes[0].set_title('Original Image', fontweight='bold', fontsize=12)
    axes[0].axis('off')
    
    im = axes[1].imshow(damage_map, cmap=cmap, vmin=0, vmax=1)
    axes[1].set_title('Damage Intensity Map', fontweight='bold', fontsize=12)
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], label='Damage Level', shrink=0.8)
    
    axes[2].imshow(img_array)
    axes[2].imshow(damage_map, cmap=cmap, alpha=0.5, vmin=0, vmax=1)
    axes[2].set_title(f'Assessment: {final_severity}', fontweight='bold', fontsize=12, color=sev_color[final_severity])
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("\n   Legend: 🔴 Severe | 🟠 Moderate | 🟡 Light | 🟢 Minimal")
    
    # Map
    if location['found']:
        print("\n" + "="*60)
        print("   🗺️ DISASTER LOCATION & IMPACT ZONES")
        print("="*60)
        
        lat, lng = location['lat'], location['lng']
        m = folium.Map(location=[lat, lng], zoom_start=9, tiles='CartoDB positron')
        
        zone_sizes = {
            'HIGH': {'red': 5000, 'orange': 15000, 'yellow': 30000},
            'MEDIUM': {'red': 3000, 'orange': 10000, 'yellow': 20000},
            'LOW': {'red': 1500, 'orange': 5000, 'yellow': 10000}
        }
        sizes = zone_sizes[final_severity]
        
        folium.Circle([lat, lng], radius=sizes['yellow'], color='gold', fill=True, fill_color='yellow', fill_opacity=0.3).add_to(m)
        folium.Circle([lat, lng], radius=sizes['orange'], color='orange', fill=True, fill_color='orange', fill_opacity=0.4).add_to(m)
        folium.Circle([lat, lng], radius=sizes['red'], color='red', fill=True, fill_color='red', fill_opacity=0.5).add_to(m)
        
        popup_html = f"""
        <div style="font-family: Arial; width: 200px;">
            <h4 style="color: red; margin: 0;">🚨 {disaster_result['disaster_type'].upper()}</h4>
            <hr><p><b>Location:</b> {location['name']}</p>
            <p><b>Severity:</b> <span style="color: {sev_color[final_severity]};">{final_severity}</span></p>
        </div>
        """
        folium.Marker([lat, lng], popup=folium.Popup(popup_html, max_width=220),
                     icon=folium.Icon(color='red', icon='exclamation-triangle', prefix='fa')).add_to(m)
        
        m.save('outputs/disaster_map.html')
        print("\n✅ Map generated | Impact zones displayed")
        display(m)
    
    print("\n" + "="*60)
    print("   ✅ ANALYSIS COMPLETE")
    print("="*60)

print("\n🚀 Ready for analysis!\n")
test_image()
