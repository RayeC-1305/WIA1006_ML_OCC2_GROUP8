import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import joblib
import os
import sys

# Add src to path so we can import our custom modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.feature_engineering import add_engineered_features

CLASS_NAMES = ["Negative", "Neutral", "Positive"]

class ModernPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Tying the (Data) Knot")
        self.root.geometry("600x600") 
        self.root.configure(bg="#FFFFFF") # Clean white background

        # Colors & Fonts
        self.BG = "#FFFFFF"
        self.FG = "#1C1C1E"
        self.ACCENT = "#000000" # Sleek black accent
        self.BTN_FG = "#FFFFFF"
        self.FONT = ("Segoe UI", 10)
        self.FONT_BOLD = ("Segoe UI", 10, "bold")
        self.TITLE_FONT = ("Segoe UI", 18, "bold")
        self.SUBTITLE_FONT = ("Segoe UI", 12)

        # Style Configuration
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TFrame', background=self.BG)
        style.configure('TLabel', background=self.BG, foreground=self.FG, font=self.FONT)
        style.configure('TCombobox', fieldbackground="#F2F2F7", background="#FFFFFF", font=self.FONT, borderwidth=0)
        style.configure('TEntry', fieldbackground="#F2F2F7", font=self.FONT, borderwidth=0, padding=5)

        # Load Model
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "xgboost.pkl")
        if not os.path.exists(model_path):
            messagebox.showerror("Error", "Could not find models/xgboost.pkl! Run main.py first.")
            self.root.destroy()
            return
            
        self.model = joblib.load(model_path)

        self._build_ui()

    def _build_ui(self):
        # --- SCROLLABLE CONTAINER SETUP ---
        # 1. Main container holding canvas and scrollbar
        container = tk.Frame(self.root, bg=self.BG)
        container.pack(fill=tk.BOTH, expand=True)

        # 2. Canvas
        canvas = tk.Canvas(container, bg=self.BG, highlightthickness=0)
        
        # 3. Scrollbar
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        
        # 4. Scrollable Frame inside Canvas
        self.scrollable_frame = tk.Frame(canvas, bg=self.BG, padx=30, pady=20)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw", width=canvas.winfo_width())
        
        # Make the inner frame expand to canvas width
        def configure_canvas_width(event):
            canvas.itemconfig(canvas.find_withtag("all")[0], width=event.width)
        canvas.bind("<Configure>", configure_canvas_width)

        canvas.configure(yscrollcommand=scrollbar.set)

        # Add mouse wheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        # --- END SCROLLABLE CONTAINER SETUP ---

        # Headers
        tk.Label(self.scrollable_frame, text="Match Predictor", font=self.TITLE_FONT, bg=self.BG, fg=self.FG, anchor="w").pack(fill=tk.X)
        tk.Label(self.scrollable_frame, text="AI Analysis Dashboard", font=self.SUBTITLE_FONT, bg=self.BG, fg="#8E8E93", anchor="w").pack(fill=tk.X, pady=(0, 20))

        # Variables
        self.vars = {
            "gender": tk.StringVar(value="Female"),
            "age": tk.IntVar(value=25),
            "location_type": tk.StringVar(value="Urban"),
            "app_usage_time_min": tk.IntVar(value=45),
            "swipe_right_ratio": tk.DoubleVar(value=50.0),
            "swipe_time_of_day": tk.StringVar(value="Evening"),
            "last_active_hour": tk.IntVar(value=20),
            "profile_pics_count": tk.IntVar(value=4),
            "bio_length": tk.IntVar(value=150),
            "interest_tags": tk.StringVar(value="Music, Travel, Coffee"),
            "likes_received": tk.IntVar(value=100),
            "message_sent_count": tk.IntVar(value=50),
            "emoji_usage_rate": tk.DoubleVar(value=20.0),
            "income_bracket": tk.StringVar(value="Middle"),
            "education_level": tk.StringVar(value="Bachelor's")
        }

        # Form Container (Return to 1 clean column since we can scroll)
        form_frame = tk.Frame(self.scrollable_frame, bg=self.BG)
        form_frame.pack(fill=tk.BOTH, expand=True)

        def add_input(label_text, var, input_type="entry", options=None):
            row_frame = tk.Frame(form_frame, bg=self.BG)
            row_frame.pack(fill=tk.X, pady=8)
            
            tk.Label(row_frame, text=label_text, font=self.FONT, bg=self.BG, fg=self.FG, anchor="w", width=25).pack(side=tk.LEFT)
            
            if input_type == "combo":
                cb = ttk.Combobox(row_frame, textvariable=var, values=options, state="readonly")
                cb.pack(side=tk.RIGHT, fill=tk.X, expand=True)
            else:
                entry = ttk.Entry(row_frame, textvariable=var)
                entry.pack(side=tk.RIGHT, fill=tk.X, expand=True)

        add_input("Gender", self.vars["gender"], "combo", ["Male", "Female", "Non-binary", "Other"])
        add_input("Age", self.vars["age"])
        add_input("Location", self.vars["location_type"], "combo", ["Urban", "Suburban", "Rural"])
        add_input("Income", self.vars["income_bracket"], "combo", ["Low", "Middle", "High"])
        add_input("Education", self.vars["education_level"], "combo", ["High School", "Bachelor's", "Master's", "PhD"])
        add_input("Daily App Usage (Mins)", self.vars["app_usage_time_min"])
        add_input("Swipe Right Ratio (%)", self.vars["swipe_right_ratio"])
        add_input("Swipe Time", self.vars["swipe_time_of_day"], "combo", ["Morning", "Afternoon", "Evening", "Night"])
        add_input("Last Active Hour (0-23)", self.vars["last_active_hour"])
        add_input("Profile Pics", self.vars["profile_pics_count"])
        add_input("Bio Length (Chars)", self.vars["bio_length"])
        add_input("Interests (comma sep)", self.vars["interest_tags"])
        add_input("Likes Received", self.vars["likes_received"])
        add_input("Messages Sent", self.vars["message_sent_count"])
        add_input("Emoji Usage (%)", self.vars["emoji_usage_rate"])

        # Minimalist Predict Button Wrapper
        btn_frame = tk.Frame(self.scrollable_frame, bg=self.BG)
        btn_frame.pack(fill=tk.X, pady=(30, 20))
        
        predict_btn = tk.Button(
            btn_frame, 
            text="PREDICT OUTCOME", 
            font=self.FONT_BOLD, 
            bg=self.ACCENT, 
            fg=self.BTN_FG, 
            relief="flat", 
            borderwidth=0,
            cursor="hand2",
            command=self.predict,
            height=2
        )
        predict_btn.pack(fill=tk.X)

        def on_enter(e):
            predict_btn['background'] = '#333333'
        def on_leave(e):
            predict_btn['background'] = self.ACCENT
            
        predict_btn.bind("<Enter>", on_enter)
        predict_btn.bind("<Leave>", on_leave)

    def predict(self):
        try:
            input_dict = {
                "gender": self.vars["gender"].get(),
                "age": self.vars["age"].get(),
                "location_type": self.vars["location_type"].get(),
                "app_usage_time_min": self.vars["app_usage_time_min"].get(),
                "swipe_right_ratio": self.vars["swipe_right_ratio"].get() / 100.0,
                "swipe_time_of_day": self.vars["swipe_time_of_day"].get(),
                "last_active_hour": self.vars["last_active_hour"].get(),
                "profile_pics_count": self.vars["profile_pics_count"].get(),
                "bio_length": self.vars["bio_length"].get(),
                "interest_tags": self.vars["interest_tags"].get(),
                "likes_received": self.vars["likes_received"].get(),
                "message_sent_count": self.vars["message_sent_count"].get(),
                "emoji_usage_rate": self.vars["emoji_usage_rate"].get() / 100.0,
                "income_bracket": self.vars["income_bracket"].get(),
                "education_level": self.vars["education_level"].get(),
            }

            df = pd.DataFrame([input_dict])
            df = add_engineered_features(df)

            pred_idx = self.model.predict(df)[0]
            probs = self.model.predict_proba(df)[0]
            outcome = CLASS_NAMES[pred_idx] if hasattr(self.model, 'classes_') else ["Negative", "Neutral", "Positive"][pred_idx]
            
            prob_text = f"Positive: {probs[2]*100:.1f}%  |  Neutral: {probs[1]*100:.1f}%  |  Negative: {probs[0]*100:.1f}%"
            
            self._show_custom_popup(outcome, prob_text)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to predict.\n{e}")

    def _show_custom_popup(self, outcome, prob_text):
        popup = tk.Toplevel(self.root)
        popup.title("Analysis Complete")
        popup.configure(bg=self.BG, padx=30, pady=20)
        popup.resizable(False, False)
        
        popup.transient(self.root)
        popup.grab_set()

        colors = {"Positive": "#34C759", "Neutral": "#FFCC00", "Negative": "#FF3B30"}
        color = colors.get(outcome, self.FG)

        tk.Label(popup, text="Match Prediction", font=self.SUBTITLE_FONT, bg=self.BG, fg="#8E8E93").pack(pady=(0, 5))
        tk.Label(popup, text=outcome.upper(), font=("Segoe UI", 28, "bold"), bg=self.BG, fg=color).pack()
        
        tk.Frame(popup, bg="#E5E5EA", height=1).pack(fill=tk.X, pady=15) 
        
        tk.Label(popup, text="Confidence Probabilities:", font=self.FONT_BOLD, bg=self.BG, fg=self.FG).pack()
        tk.Label(popup, text=prob_text, font=self.FONT, bg=self.BG, fg=self.FG).pack(pady=(5, 20))
        
        tk.Button(popup, text="DONE", font=self.FONT_BOLD, bg="#000000", fg="#FFFFFF", relief="flat", borderwidth=0, cursor="hand2", command=popup.destroy, width=20, height=2).pack()

if __name__ == "__main__":
    root = tk.Tk()
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except:
        pass
    app = ModernPredictorApp(root)
    root.mainloop()
