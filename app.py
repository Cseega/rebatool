import cv2
import av
import numpy as np
import streamlit as st
import pandas as pd
import tempfile
from streamlit_webrtc import webrtc_streamer
import mediapipe as mp

# --- 1. BEÁLLÍTÁSOK ÉS ÁLLAPOT (STATE) ---
st.set_page_config(page_title="Ergonómiai Elemző", layout="wide")

if "log_data" not in st.session_state:
    st.session_state.log_data = []

# ÚJ: REBA Módosítók alapértelmezett értékei
if "method" not in st.session_state: st.session_state.method = "REBA"
if "load_score" not in st.session_state: st.session_state.load_score = 0
if "coupling_score" not in st.session_state: st.session_state.coupling_score = 0
if "activity_score" not in st.session_state: st.session_state.activity_score = 0

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.7,  # Szigorítottuk 70%-ra a fals felismerések ellen
    min_tracking_confidence=0.7    # Szigorítottuk 70%-ra
)

# --- 2. MATEMATIKAI ÉS ERGONÓMIAI FÜGGVÉNYEK ---
def calculate_angle(a, b, c):
    """Kiszámolja az a-b-c pontok által bezárt szöget 3D TÉRBEN."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    # Koszinusz tétel vektorokra
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def calculate_vertical_angle(a, b):
    """Kiszámolja a függőlegeshez képesti dőlést 3D TÉRBEN."""
    a, b = np.array(a), np.array(b)
    vector = a - b
    # 3D függőleges referencia vektor (Y tengely mutat lefelé a képen)
    vertical = np.array([0, -1, 0]) 
    norm_v = np.linalg.norm(vector)
    if norm_v == 0: return 0
    cosine_angle = np.dot(vector, vertical) / norm_v
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)
    
def get_color(score, max_score):
    """Visszaad egy BGR színt a kapott ergonómiai pontszám alapján (Zöld -> Sárga -> Narancs -> Piros)."""
    ratio = score / max_score
    if ratio <= 0.35: return (0, 255, 0)      # Zöld: Biztonságos
    elif ratio <= 0.65: return (0, 255, 255)  # Sárga: Figyelmeztetés
    elif ratio <= 0.85: return (0, 165, 255)  # Narancs: Magas kockázat
    else: return (0, 0, 255)                  # Piros: Veszélyes

# --- SZABVÁNYOS REBA TÁBLÁZATOK (MÁTRIXOK) ---
def get_reba_table_a(trunk, neck, leg):
    """REBA 'A' táblázat (Törzs, Nyak, Láb). Visszaadja az A pontszámot."""
    # Indexek: Törzs (1-5), Nyak (1-3), Láb (1-4)
    table_a = [
        [[1,2,3,4], [2,3,4,5], [3,3,5,6]], # Törzs 1
        [[2,3,4,5], [3,4,5,6], [4,5,6,7]], # Törzs 2
        [[3,4,5,6], [4,5,6,7], [5,6,7,8]], # Törzs 3
        [[4,5,6,7], [5,6,7,8], [6,7,8,9]], # Törzs 4
        [[5,6,7,8], [6,7,8,9], [7,8,9,9]]  # Törzs 5
    ]
    t = max(0, min(trunk-1, 4))
    n = max(0, min(neck-1, 2))
    l = max(0, min(leg-1, 3))
    return table_a[t][n][l]

def get_reba_table_b(upper_arm, lower_arm, wrist):
    """REBA 'B' táblázat (Felkar, Alkar, Csukló). Visszaadja a B pontszámot."""
    # Indexek: Felkar (1-6), Alkar (1-2), Csukló (1-3)
    table_b = [
        [[1,2,2], [1,2,3]], # Felkar 1
        [[1,2,3], [2,3,4]], # Felkar 2
        [[3,4,5], [4,5,5]], # Felkar 3
        [[4,5,5], [5,6,7]], # Felkar 4
        [[6,7,8], [7,8,8]], # Felkar 5
        [[7,8,8], [8,9,9]]  # Felkar 6
    ]
    u = max(0, min(upper_arm-1, 5))
    la = max(0, min(lower_arm-1, 1))
    w = max(0, min(wrist-1, 2))
    return table_b[u][la][w]

def get_reba_table_c(score_a, score_b):
    """REBA 'C' táblázat. Visszaadja a végső REBA alapértéket (1-15)."""
    table_c = [
        [1,1,1,2,3,3,4,5,6,7,7,7],
        [1,2,2,3,4,4,5,6,6,7,7,8],
        [2,3,3,3,4,5,6,7,7,8,8,8],
        [3,4,4,4,5,6,7,8,8,9,9,9],
        [4,4,4,5,6,7,8,8,9,9,9,9],
        [6,6,6,7,8,8,9,9,10,10,10,10],
        [7,7,7,8,9,9,9,10,10,11,11,11],
        [8,8,8,9,10,10,10,10,10,11,11,11],
        [9,9,9,10,10,10,11,11,11,12,12,12],
        [10,10,10,11,11,11,11,12,12,12,12,12],
        [11,11,11,11,12,12,12,12,12,12,12,12],
        [12,12,12,12,12,12,12,12,12,12,12,12]
    ]
    a = max(0, min(score_a-1, 11))
    b = max(0, min(score_b-1, 11))
    return table_c[a][b]

# --- SZABVÁNYOS RULA TÁBLÁZATOK ---
def get_rula_table_a(upper, lower, wrist, twist=1):
    """RULA 'A' tábla (Karok és Csukló)."""
    table = [
        [[[1,2],[2,2],[2,3],[3,3]], [[2,2],[2,2],[3,3],[3,3]], [[2,3],[3,3],[3,3],[4,4]]],
        [[[2,2],[2,2],[3,3],[3,3]], [[2,2],[2,2],[3,3],[3,3]], [[2,3],[3,3],[3,4],[4,4]]],
        [[[2,3],[3,3],[3,4],[4,4]], [[2,3],[3,3],[3,4],[4,4]], [[3,3],[3,4],[4,4],[5,5]]],
        [[[3,3],[3,3],[4,4],[4,4]], [[3,3],[3,3],[4,4],[4,4]], [[4,4],[4,4],[4,5],[5,5]]],
        [[[4,4],[4,4],[4,5],[5,5]], [[4,4],[4,4],[4,5],[5,5]], [[4,4],[4,5],[5,5],[6,6]]],
        [[[5,5],[5,5],[5,6],[6,7]], [[5,5],[5,5],[5,6],[6,7]], [[5,5],[5,6],[6,6],[7,7]]]
    ]
    u, l = max(0, min(upper-1, 5)), max(0, min(lower-1, 2))
    w, t = max(0, min(wrist-1, 3)), max(0, min(twist-1, 1))
    return table[u][l][w][t]

def get_rula_table_b(neck, trunk, leg):
    """RULA 'B' tábla (Nyak, Törzs, Láb)."""
    table = [
        [[1,3], [2,3], [3,4], [5,5], [6,6], [7,7]],
        [[2,3], [2,3], [4,5], [5,5], [6,7], [7,7]],
        [[3,3], [3,4], [4,5], [5,6], [6,7], [7,7]],
        [[5,5], [5,6], [5,6], [6,7], [7,7], [7,7]],
        [[7,7], [7,7], [7,7], [7,7], [7,7], [7,7]],
        [[7,7], [7,7], [7,7], [7,7], [7,7], [7,7]]
    ]
    n, t, lg = max(0, min(neck-1, 5)), max(0, min(trunk-1, 5)), max(0, min(leg-1, 1))
    return table[n][t][lg]

def get_rula_table_c(score_a, score_b):
    """RULA Végső 'C' tábla."""
    table = [
        [1,2,3,3,4,5,5], [2,2,3,4,4,5,5], [3,3,3,4,4,5,6], [3,3,3,4,5,6,6],
        [4,4,4,5,6,7,7], [4,4,5,6,6,7,7], [5,5,6,6,7,7,7], [5,5,6,7,7,7,7]
    ]
    a, b = max(0, min(score_a-1, 7)), max(0, min(score_b-1, 6))
    return table[a][b]

# --- 3. KÉPKOCKA FELDOLGOZÓ MOTOR ---
def process_frame_data(img, frame_index=0, load_score=0, coupling_score=0, activity_score=0, method="REBA", score_history=None):
    h, w, _ = img.shape
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    out_img = img.copy()
    log_row = None 
    
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        mp_drawing.draw_landmarks(out_img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # 3D mélység beemelése
        def get_pt(idx):
            return np.array([landmarks[idx].x * w, landmarks[idx].y * h, landmarks[idx].z * w])
            
        def is_vis(*indices, threshold=0.5):
            return all(landmarks[i].visibility > threshold for i in indices)

        # --- GLITCH SZŰRŐ (Törzs) ---
        if not is_vis(11, 12, 23, 24, threshold=0.5):
            cv2.rectangle(out_img, (0,0), (450, 60), (0, 0, 0), -1)
            cv2.putText(out_img, "MERES SZUNETEL: Takarasban", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            return out_img, None 

        # --- DINAMIKUS KULCSPONTOK ---
        l_ear, r_ear = get_pt(7), get_pt(8)
        l_shoulder, r_shoulder = get_pt(11), get_pt(12)
        l_hip, r_hip = get_pt(23), get_pt(24)
        l_elbow, r_elbow = get_pt(13), get_pt(14)
        l_wrist, r_wrist = get_pt(15), get_pt(16)
        l_index, r_index = get_pt(19), get_pt(20)
        l_knee, r_knee = get_pt(25), get_pt(26)
        l_ankle, r_ankle = get_pt(27), get_pt(28)

        head_mid = (l_ear + r_ear) / 2
        shoulder_mid = (l_shoulder + r_shoulder) / 2
        hip_mid = (l_hip + r_hip) / 2

        is_rula = "RULA" in method
        method_name = "RULA" if is_rula else "REBA"
        
        trunk_twist = abs(landmarks[11].z - landmarks[12].z) > 0.15 or abs(landmarks[11].y - landmarks[12].y) > 0.05
        neck_twist = is_vis(7, 8) and (abs(landmarks[7].z - landmarks[8].z) > 0.15 or abs(landmarks[7].y - landmarks[8].y) > 0.05)

        # --- KÖZÖS LÁTHATÓSÁG ÉS SZÖGEK ---
        l_leg_vis = is_vis(23, 25, 27, threshold=0.7)
        r_leg_vis = is_vis(24, 26, 28, threshold=0.7)
        trunk_angle = calculate_vertical_angle(shoulder_mid, hip_mid)

        # --- DINAMIKUS PONTOZÁS ---
        if is_rula:
            # RULA: Törzs és Nyak
            trunk_score = 1 if trunk_angle <= 10 else (2 if trunk_angle <= 20 else (3 if trunk_angle <= 60 else 4))
            if trunk_twist: trunk_score = min(trunk_score + 1, 5)

            if is_vis(7, 8):
                neck_angle = calculate_vertical_angle(head_mid, shoulder_mid)
                neck_score = 1 if neck_angle <= 10 else (2 if neck_angle <= 20 else 3)
                if neck_twist: neck_score = min(neck_score + 1, 4)
            else: neck_score = 1

            # RULA: Lábak szigorú szűrővel
            leg_score = 1 
            if l_leg_vis and r_leg_vis:
                l_knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
                r_knee_angle = calculate_angle(r_hip, r_knee, r_ankle)
                leg_score = 1 if min(l_knee_angle, r_knee_angle) >= 150 else 2
            elif l_leg_vis or r_leg_vis:
                leg_score = 2 
            else:
                cv2.rectangle(out_img, (0,0), (450, 60), (0, 0, 0), -1)
                cv2.putText(out_img, "LABAK TAKARASBAN", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                return out_img, None

            # RULA: Karok
            def get_rula_arm_scores(shoulder, elbow, wrist, index, hip):
                u_ang = calculate_vertical_angle(shoulder, elbow)
                u_sc = 1 if u_ang <= 20 else (2 if u_ang <= 45 else (3 if u_ang <= 90 else 4))
                if calculate_angle(hip, shoulder, elbow) > 30: u_sc = min(u_sc + 1, 6) 
                l_ang = calculate_angle(shoulder, elbow, wrist)
                l_sc = 1 if 60 <= l_ang <= 100 else 2
                w_ang = calculate_angle(elbow, wrist, index)
                w_sc = 1 if w_ang >= 170 else (2 if w_ang >= 150 else 3)
                return u_sc, l_sc, w_sc, u_ang

            l_u, l_l, l_w, l_upper_angle = 1, 1, 1, 0
            if is_vis(11, 13, 15, 23): l_u, l_l, l_w, l_upper_angle = get_rula_arm_scores(l_shoulder, l_elbow, l_wrist, l_index, l_hip)
            r_u, r_l, r_w, r_upper_angle = 1, 1, 1, 0
            if is_vis(12, 14, 16, 24): r_u, r_l, r_w, r_upper_angle = get_rula_arm_scores(r_shoulder, r_elbow, r_wrist, r_index, r_hip)

            upper_arm_score, lower_arm_score, wrist_score = max(l_u, r_u), max(l_l, r_l), max(l_w, r_w)
            upper_arm_angle = l_upper_angle if l_u > r_u else r_upper_angle

            score_a_base = get_rula_table_a(upper_arm_score, lower_arm_score, wrist_score, 1)
            score_a = score_a_base + activity_score + load_score
            score_b_base = get_rula_table_b(neck_score, trunk_score, leg_score)
            score_b = score_b_base + activity_score + load_score
            raw_final_score = get_rula_table_c(score_a, score_b)
            max_possible = 7

        else:
            # REBA: Törzs és Nyak
            trunk_score = 1 if trunk_angle <= 20 else (3 if trunk_angle <= 60 else 4)
            if trunk_twist: trunk_score = min(trunk_score + 1, 5)

            if is_vis(7, 8):
                neck_angle = calculate_vertical_angle(head_mid, shoulder_mid)
                neck_score = 1 if neck_angle <= 20 else 2
                if neck_twist: neck_score = min(neck_score + 1, 3)
            else: neck_score = 1

            # REBA: Lábak szigorú szűrővel
            if l_leg_vis and r_leg_vis:
                worst_knee = min(calculate_angle(l_hip, l_knee, l_ankle), calculate_angle(r_hip, r_knee, r_ankle))
                leg_score = 1 if worst_knee >= 150 else (2 if worst_knee >= 120 else 3)
            elif l_leg_vis or r_leg_vis:
                leg_score = 2
            else:
                cv2.rectangle(out_img, (0,0), (450, 60), (0, 0, 0), -1)
                cv2.putText(out_img, "LABAK TAKARASBAN", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                return out_img, None

            # REBA: Karok
            def get_reba_arm_scores(shoulder, elbow, wrist, index, hip):
                u_ang = calculate_vertical_angle(shoulder, elbow)
                u_sc = 1 if u_ang <= 20 else (2 if u_ang <= 45 else (3 if u_ang <= 90 else 4))
                if calculate_angle(hip, shoulder, elbow) > 30: u_sc = min(u_sc + 1, 6)
                l_ang = calculate_angle(shoulder, elbow, wrist)
                l_sc = 1 if 60 <= l_ang <= 100 else 2
                w_ang = calculate_angle(elbow, wrist, index)
                w_sc = 1 if w_ang >= 165 else 2
                return u_sc, l_sc, w_sc, u_ang

            l_u, l_l, l_w, l_upper_angle = 1, 1, 1, 0
            if is_vis(11, 13, 15, 23): l_u, l_l, l_w, l_upper_angle = get_reba_arm_scores(l_shoulder, l_elbow, l_wrist, l_index, l_hip)
            r_u, r_l, r_w, r_upper_angle = 1, 1, 1, 0
            if is_vis(12, 14, 16, 24): r_u, r_l, r_w, r_upper_angle = get_reba_arm_scores(r_shoulder, r_elbow, r_wrist, r_index, r_hip)

            upper_arm_score, lower_arm_score, wrist_score = max(l_u, r_u), max(l_l, r_l), max(l_w, r_w)
            upper_arm_angle = l_upper_angle if l_u > r_u else r_upper_angle

            score_a_base = get_reba_table_a(trunk_score, neck_score, leg_score)
            score_a = score_a_base + load_score
            score_b_base = get_reba_table_b(upper_arm_score, lower_arm_score, wrist_score)
            score_b = score_b_base + coupling_score
            raw_final_score = get_reba_table_c(score_a, score_b) + activity_score
            max_possible = 15

        # --- IDŐBELI SIMÍTÁS (MOVING AVERAGE) ---
        if score_history is not None:
            score_history.append(raw_final_score)
            if len(score_history) > 5: # Az utolsó 5 képkockát veszi figyelembe
                score_history.pop(0)
            final_score = int(round(sum(score_history) / len(score_history))) # Átlagolás a stabilitásért
        else:
            final_score = raw_final_score

        # --- TELJES BIOMECHANIKAI ADATNAPLÓZÁS ---
        # Kimentjük mindkét oldal adatait, hogy az Excelben minden visszakövethető legyen!
        log_row = {
            "Frame": frame_index,
            "Végső_Pontszám": final_score,
            "Törzs_szög (°)": round(trunk_angle, 1),
            "Nyak_szög (°)": round(neck_angle, 1) if is_vis(7, 8) else "Takarásban",
            "Bal_Térd (°)": round(calculate_angle(l_hip, l_knee, l_ankle), 1) if l_leg_vis else "Takarásban",
            "Jobb_Térd (°)": round(calculate_angle(r_hip, r_knee, r_ankle), 1) if r_leg_vis else "Takarásban",
            "Bal_Felkar (°)": round(calculate_vertical_angle(l_shoulder, l_elbow), 1) if is_vis(11, 13) else "Takarásban",
            "Jobb_Felkar (°)": round(calculate_vertical_angle(r_shoulder, r_elbow), 1) if is_vis(12, 14) else "Takarásban",
            "Bal_Könyök (°)": round(calculate_angle(l_shoulder, l_elbow, l_wrist), 1) if is_vis(11, 13, 15) else "Takarásban",
            "Jobb_Könyök (°)": round(calculate_angle(r_shoulder, r_elbow, r_wrist), 1) if is_vis(12, 14, 16) else "Takarásban",
            "A_Csoport_Pont": score_a,
            "B_Csoport_Pont": score_b
        }

        # --- VIZUÁLIS SZÍNKÓDOLÁS ---
        def draw_bone(p1, p2, score, m_score):
            pt1 = (int(p1[0]), int(p1[1]))
            pt2 = (int(p2[0]), int(p2[1]))
            cv2.line(out_img, pt1, pt2, get_color(score, m_score), 6)

        # SZABVÁNYOS MAXIMUMOK (Hogy a színek a valós kockázatot mutassák)
        max_trunk = 4 if is_rula else 5
        max_neck = 4 if is_rula else 3
        max_arm = 6 # Mindkét szabványnál felmehet 6-ig az abdukció/vállhúzás miatt
        max_lower = 2
        max_leg = 2 if is_rula else 4 # REBA-nál a láb 4-ig mehet, RULA-nál csak 2-ig!

        draw_bone(shoulder_mid, hip_mid, trunk_score, max_trunk)
        if is_vis(7, 8): draw_bone(head_mid, shoulder_mid, neck_score, max_neck)
        
        if is_vis(11, 13): draw_bone(l_shoulder, l_elbow, l_u, max_arm)
        if is_vis(12, 14): draw_bone(r_shoulder, r_elbow, r_u, max_arm)
        if is_vis(13, 15): draw_bone(l_elbow, l_wrist, l_l, max_lower)
        if is_vis(14, 16): draw_bone(r_elbow, r_wrist, r_l, max_lower)
        
        # A lábak most már a helyes maximumhoz viszonyítva lesznek színezve
        if is_vis(23, 25): draw_bone(l_hip, l_knee, leg_score, max_leg)
        if is_vis(24, 26): draw_bone(r_hip, r_knee, leg_score, max_leg)
        if is_vis(25, 27): draw_bone(l_knee, l_ankle, leg_score, max_leg)
        if is_vis(26, 28): draw_bone(r_knee, r_ankle, leg_score, max_leg)

        cv2.rectangle(out_img, (0,0), (320, 100), (0, 0, 0), -1)
        cv2.putText(out_img, f"{method_name} SCORE: {final_score}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.2, get_color(final_score, max_possible), 3)
        cv2.putText(out_img, f"Group A: {score_a} | Group B: {score_b}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # --- VESZÉLYESSÉGI RIASZTÁS ---
        alert_threshold = 6 if is_rula else 11
        if final_score >= alert_threshold:
            if frame_index % 10 < 5: 
                cv2.rectangle(out_img, (0, 0), (w, h), (0, 0, 255), 20)
                text = "VESZELYES POZICIO!"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
                text_x = (w - text_size[0]) // 2
                cv2.putText(out_img, text, (text_x, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    return out_img, log_row

# --- WebRTC Processor (Élőképhez) ---
class RebaLiveProcessor:
    def __init__(self):
        self.frame_count = 0
        self.load_score = 0
        self.coupling_score = 0
        self.activity_score = 0
        self.method = "REBA" 
        self.live_data = [] 
        self.score_history = [] # Simításhoz (Live)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1
        
        processed_img, log_row = process_frame_data(
            img, 
            self.frame_count, 
            load_score=self.load_score,
            coupling_score=self.coupling_score,
            activity_score=self.activity_score,
            method=self.method,
            score_history=self.score_history # Átadjuk a memóriát
        )
        
        if log_row is not None:
            self.live_data.append(log_row)
            
        return av.VideoFrame.from_ndarray(processed_img, format="bgr24")

# --- 4. STREAMLIT FELÜLET (UI) ---
st.title("Komplex Ergonómiai Elemző (REBA & RULA Szabvány)")
st.write("A videóból vagy kameraképből kinyert adatok lentebb Excelbe menthetők.")

with st.sidebar:
    st.header("⚙️ Elemzési Módszer")
    st.session_state.method = st.radio(
        "Válassz szabványt:",
        options=["REBA (Teljes test)", "RULA (Felső végtag)"],
        help="A REBA fizikai munkához, a RULA ülő/finommotoros munkához ideális."
    )
    st.markdown("---")

    st.header("⚙️ Módosítók")
    st.info("Ezeket a kamera nem látja, manuálisan kell beállítani a munkakörnyezet alapján.")
    
    st.session_state.load_score = st.selectbox(
        "📦 Teher / Erő (Load/Force)", 
        options=[0, 1, 2], 
        format_func=lambda x: ["0: < 5 kg", "1: 5 - 10 kg", "2: > 10 kg (vagy hirtelen)"][x]
    )
    
    st.session_state.coupling_score = st.selectbox(
        "🤝 Fogás minősége (Coupling)", 
        options=[0, 1, 2, 3], 
        format_func=lambda x: ["0: Jó fogás (fogantyú)", "1: Elfogadható", "2: Gyenge fogás", "3: Veszélyes"][x]
    )
    
    st.session_state.activity_score = st.selectbox(
        "⏱️ Tevékenység (Activity)", 
        options=[0, 1], 
        format_func=lambda x: ["0: Normál mozgás", "1: Statikus VAGY Ismétlődő"][x]
    )

tab1, tab2 = st.tabs(["Élőkép (Kamera)", "Videó feltöltése"])

with tab1:
    st.header("Valós idejű elemzés")
    ctx = webrtc_streamer(
        key="reba-live", 
        video_processor_factory=RebaLiveProcessor,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    
    if ctx.video_processor:
        ctx.video_processor.load_score = st.session_state.load_score
        ctx.video_processor.coupling_score = st.session_state.coupling_score
        ctx.video_processor.activity_score = st.session_state.activity_score
        ctx.video_processor.method = st.session_state.method
        
        st.markdown("### Élő mérés mentése")
        st.info("A kamera a háttérben rögzíti az adatokat. Kattints a gombra, hogy átkerüljenek a grafikonba és az Excelbe!")
        
        if st.button("📥 Élőkép adatainak átvezetése a grafikonba", use_container_width=True):
            if len(ctx.video_processor.live_data) > 0:
                st.session_state.log_data.extend(ctx.video_processor.live_data)
                ctx.video_processor.live_data = [] 
                st.rerun() 
            else:
                st.warning("Még nincs rögzített adat (nem látszott ember a kamerában).")

with tab2:
    st.header("Utólagos videóelemzés")
    uploaded_file = st.file_uploader("Tölts fel egy videót (mp4, mov)", type=["mp4", "mov"])
    
    if uploaded_file is not None:
        import tempfile
        import os
        
        # Kiolvassuk a bytokat a memóriába, hogy többször is felhasználhassuk
        file_bytes = uploaded_file.read()
        
        # --- 1. JAVÍTÁS: Nyers előnézet (Hogyan látja az OpenCV) ---
        col_prev1, col_prev2 = st.columns(2)
        with col_prev1:
            st.markdown("**1. Böngésző előnézete:**")
            st.video(file_bytes)
            
        with col_prev2:
            st.markdown("**2. A program így látja (Nyers kép):**")
            tfile_prev = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") 
            tfile_prev.write(file_bytes)
            tfile_prev.close()
            
            cap_prev = cv2.VideoCapture(tfile_prev.name)
            ret_prev, frame_prev = cap_prev.read()
            cap_prev.release()
            
            if ret_prev:
                # Ezt a képet látja az AI. Ha itt fekszik, forgatni kell!
                st.image(cv2.cvtColor(frame_prev, cv2.COLOR_BGR2RGB), use_container_width=True)
                st.info("💡 Tipp: Ha a fenti 2. kép fekve van, forgasd el a lenti menüben, hogy álló legyen!")
            else:
                st.error("Nem sikerült előnézetet generálni a nyers fájlból.")
                
        st.markdown("---")
        
        # --- Forgatás opció ---
        rotation_option = st.selectbox(
            "🔄 Nyers videó elforgatása (Az OpenCV előnézet alapján)",
            ["Nincs forgatás", "90 fok jobbra", "180 fok", "90 fok balra"]
        )
        
        if st.button("Videó feldolgozása indít", use_container_width=True):
            
            tfile_in = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") 
            tfile_in.write(file_bytes) 
            tfile_in.close() 
            
            tfile_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile_out_name = tfile_out.name
            tfile_out.close()
            
            cap = cv2.VideoCapture(tfile_in.name)
            
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            if fps == 0: fps = 30 
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            v_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            v_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # --- BIZTONSÁGI ELLENŐRZÉS ---
            if total_frames == 0:
                st.error("❌ A szerver nem tudta beolvasni a videót. Valószínűleg egy Apple HEVC (H.265) formátum. Kérlek konvertáld át sima MP4-be!")
                cap.release()
            else:
                # Felbontás megfordítása, ha 90 fokkal forgatunk
                if rotation_option in ["90 fok jobbra", "90 fok balra"]:
                    v_width, v_height = v_height, v_width
                
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out_video = cv2.VideoWriter(tfile_out_name, fourcc, fps, (v_width, v_height))
                
                stframe = st.empty() 
                progress_text = st.empty()
                progress_bar = st.progress(0)
                frame_idx = 0
                
                score_history = [] 
                max_recorded_score = -1
                worst_frame_image = None
                worst_frame_idx = 0
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Képkocka fizikai forgatása
                    if rotation_option == "90 fok jobbra":
                        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                    elif rotation_option == "180 fok":
                        frame = cv2.rotate(frame, cv2.ROTATE_180)
                    elif rotation_option == "90 fok balra":
                        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    
                    frame_idx += 1
                    
                    out_frame, log_row = process_frame_data(
                        frame, 
                        frame_idx, 
                        load_score=st.session_state.load_score,
                        coupling_score=st.session_state.coupling_score,
                        activity_score=st.session_state.activity_score,
                        method=st.session_state.method,
                        score_history=score_history 
                    )
                    
                    if log_row is not None:
                        st.session_state.log_data.append(log_row)
                        
                        current_score = log_row["Végső_Pontszám"]
                        if current_score > max_recorded_score:
                            max_recorded_score = current_score
                            worst_frame_image = out_frame.copy() 
                            worst_frame_idx = frame_idx
                    
                    out_video.write(out_frame)
                    
                    # --- 2. JAVÍTÁS: Képfrissítés ritkítása és butítása a felhő miatt ---
                    # Csak minden 15. képkockát küldjük a böngészőnek (gyorsabb átvitel)
                    if frame_idx % 15 == 0:
                        preview_width = max(int(v_width / 2), 1)
                        preview_height = max(int(v_height / 2), 1)
                        preview_img = cv2.resize(out_frame, (preview_width, preview_height))
                        stframe.image(cv2.cvtColor(preview_img, cv2.COLOR_BGR2RGB), channels="RGB")
                    
                    if total_frames > 0:
                        progress = min(frame_idx / total_frames, 1.0)
                        progress_bar.progress(progress)
                        progress_text.text(f"Feldolgozás: {frame_idx} / {total_frames} képkocka ({int(progress*100)}%)")
                    
                cap.release()
                out_video.release() 
                
                st.success("Videó feldolgozása befejeződött!")
                
                col_vid, col_img = st.columns(2)
                
                with col_vid:
                    with open(tfile_out_name, "rb") as video_file:
                        video_bytes = video_file.read()
                        
                    st.download_button(
                        label="🎬 Feldolgozott videó letöltése (.mp4)",
                        data=video_bytes,
                        file_name="ergonomia_elemzes.mp4",
                        mime="video/mp4",
                        use_container_width=True
                    )
                
                # ... (itt van a legrosszabb képkocka letöltése) ...
                
                if worst_frame_image is not None:
                    st.markdown("---")
                    st.subheader(f"⚠️ Legveszélyesebb mozdulat (Képkocka: {worst_frame_idx})")
                    st.image(cv2.cvtColor(worst_frame_image, cv2.COLOR_BGR2RGB), caption=f"Maximum mért pontszám: {max_recorded_score}")
                    
                    is_success, buffer = cv2.imencode(".jpg", worst_frame_image)
                    if is_success:
                        with col_img:
                            st.download_button(
                                label="📸 Legrosszabb pillanatkép letöltése (.jpg)",
                                data=buffer.tobytes(),
                                file_name="legveszelyesebb_mozdulat.jpg",
                                mime="image/jpeg",
                                use_container_width=True
                            )
                            
                # --- ÚJ: SZEMÉTSZÁLLÍTÁS (Memória felszabadítása) ---
                # Miután mindent beolvastunk a memóriába, letöröljük a fizikai fájlokat a szerverről!
                try:
                    os.remove(tfile_in.name)
                    os.remove(tfile_out_name)
                    os.remove(tfile_prev.name)
                except Exception as e:
                    pass # Ha valamiért nem találja, csendben továbbmegy
                        
# --- 5. ADATOK EXPORTÁLÁSA ÉS VIZUALIZÁCIÓ ---
st.markdown("---")
st.subheader("📊 Elemzés eredményei és Exportálás")

if len(st.session_state.log_data) > 0:
    df = pd.DataFrame(st.session_state.log_data)
    
    # --- ERGONÓMIAI MŰSZERFAL (DASHBOARD) ---
    max_reba = int(df["Végső_Pontszám"].max())
    avg_reba = round(df["Végső_Pontszám"].mean(), 1)
    
    # Hivatalos Kockázati szintek és intézkedések (REBA és RULA)
    def get_risk_evaluation(score, method):
        if "REBA" in method:
            if score == 1: return "Elhanyagolható", "Nincs szükség beavatkozásra", "🟢"
            elif score <= 3: return "Alacsony", "Beavatkozás lehetséges", "🟡"
            elif score <= 7: return "Közepes", "Beavatkozás szükséges", "🟠"
            elif score <= 10: return "Magas", "Hamarosan beavatkozás szükséges", "🔴"
            else: return "Nagyon magas", "Azonnali beavatkozás szükséges", "🚨"
        else:
            # RULA Kockázati szintek (Max 7 pont)
            if score <= 2: return "Elfogadható", "Testtartás elfogadható", "🟢"
            elif score <= 4: return "Alacsony", "További vizsgálat javasolt", "🟡"
            elif score <= 6: return "Magas", "Hamarosan beavatkozás szükséges", "🟠"
            else: return "Nagyon magas", "Azonnali beavatkozás szükséges", "🚨"

    risk_level, action_req, icon = get_risk_evaluation(max_reba, st.session_state.method)
    
    # Három oszlop a legfontosabb statisztikáknak (Dinamikus feliratokkal)
    method_name = st.session_state.method.split()[0] # Csak a mozaikszót vesszük ki (REBA vagy RULA)
    is_rula = "RULA" in st.session_state.method
    
    dash1, dash2, dash3 = st.columns(3)
    dash1.metric(f"Mért Maximum {method_name}", f"{max_reba}")
    dash2.metric(f"Átlagos {method_name} Terhelés", f"{avg_reba}")
    dash3.info(f"**{icon} Kockázati szint:** {risk_level} \n\n **Intézkedés:** {action_req}")

    # --- Interaktív grafikon ---
    st.write(f"**{method_name} pontszám alakulása az idő (képkockák) függvényében:**")
    st.info("💡 Tipp: Vidd az egeret a grafikon fölé a pontos értékek megtekintéséhez. A csúcsok jelzik a legkockázatosabb mozdulatokat.")
    
    chart_data = df.set_index("Frame")["Végső_Pontszám"]
    st.line_chart(chart_data, color="#FF4B4B") 
    
    # --- Adattábla és Exportálás ---
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.write("**Rögzített adatok (Összes képkocka):**")
        st.dataframe(df, height=400) 
    
    with col2:
        st.write("**Adatok letöltése és törlése:**")
        
        # 1. Excel Letöltés
        @st.cache_data
        def convert_df_to_excel(df):
            import io
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Adatok')
            return output.getvalue()
        
        excel_data = convert_df_to_excel(df)
        st.download_button(
            label="📥 Részletes adatok (Excel)",
            data=excel_data,
            file_name=f'ergonomia_adatok_{method_name}.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            use_container_width=True
        )
        
        # --- ÚJ: 2. HTML / PDF Vezetői Jelentés Generálása ---
        html_report = f"""
        <html>
        <head>
            <meta charset="utf-8">
            <title>Ergonómiai Elemzési Jelentés</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; color: #333; }}
                h1 {{ color: #2C3E50; border-bottom: 2px solid #3498DB; padding-bottom: 10px; }}
                .summary-box {{ background-color: #F8F9F9; padding: 20px; border-radius: 8px; border-left: 5px solid #3498DB; margin-top: 20px; }}
                .alert {{ font-size: 18px; font-weight: bold; color: {'#E74C3C' if max_reba >= (6 if is_rula else 11) else '#27AE60'}; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
                th, td {{ border: 1px solid #BDC3C7; padding: 12px; text-align: left; }}
                th {{ background-color: #ECF0F1; }}
                .footer {{ margin-top: 50px; font-size: 12px; color: #7F8C8D; text-align: center; }}
            </style>
        </head>
        <body>
            <h1>Munkavédelmi és Ergonómiai Jelentés</h1>
            <p><strong>Dátum:</strong> Automatikusan generálva</p>
            <p><strong>Alkalmazott módszertan:</strong> {method_name} Szabvány</p>
            
            <div class="summary-box">
                <h2>Vezetői Összefoglaló</h2>
                <p>A videós elemzés alapján a dolgozó maximális ergonómiai terhelése:</p>
                <ul>
                    <li><strong>Mért Maximum {method_name} Pontszám:</strong> {max_reba} pont</li>
                    <li><strong>Átlagos Terhelés:</strong> {avg_reba} pont</li>
                    <li><strong>Kockázati Szint:</strong> {icon} {risk_level}</li>
                </ul>
                <p class="alert">Javasolt Munkavédelmi Intézkedés: {action_req}</p>
            </div>
            
            <h2>Részletes Statisztikák (Top 10 legkockázatosabb pillanat)</h2>
            {df.sort_values(by="Végső_Pontszám", ascending=False).head(10).to_html(index=False)}
            
            <div class="footer">
                Generálva a Komplex Ergonómiai Elemző szoftverrel. Kérjük, nyomtassa PDF-be az archiváláshoz!
            </div>
        </body>
        </html>
        """
        
        st.download_button(
            label="📄 Vezetői Jelentés letöltése (HTML/PDF)",
            data=html_report.encode('utf-8'),
            file_name=f'vezetői_jelentés_{method_name}.html',
            mime='text/html',
            use_container_width=True
        )
        
        # 3. Adatok Törlése
        st.markdown("---")
        if st.button("🗑️ Rögzített adatok törlése (Új mérés kezdése)", use_container_width=True):
            st.session_state.log_data = []
            st.rerun()

else:
    st.info("Még nincsenek rögzített adatok. Indítsd el a kamerát vagy tölts fel és dolgozz fel egy videót.")
