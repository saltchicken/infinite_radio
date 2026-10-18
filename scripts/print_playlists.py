import sqlite3
import json

db_path = "../votes.db"

def print_playlists():
    try:
        # Connect to the database
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name, active_style, abc, abc_order, cot FROM playlists")
            rows = cursor.fetchall()
            
            if not rows:
                print("No playlists found in the database.")
                return
            
            print("=== Your Saved Playlists ===\n")
            
            # Loop through and format each playlist
            for row in rows:
                name = row[0]
                styles = json.loads(row[1]) if row[1] else []
                abcs = json.loads(row[2]) if row[2] else []
                order = row[3]
                cot = row[4]
                
                print(f"🎵 Playlist: {name}")
                print(f"   Order: {order.title()} | CoT Mode: {cot}")
                
                print(f"   Styles ({len(styles)}):")
                if not styles:
                    print("      (None)")
                for style in styles:
                    print(f"      - {style}")
                    
                print(f"   ABC Scores ({len(abcs)}):")
                if not abcs:
                    print("      (None)")
                for abc in abcs:
                    print(f"      - {abc}")
                    
                print("-" * 40)
                
    except sqlite3.Error as e:
        print(f"Database error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    print_playlists()
