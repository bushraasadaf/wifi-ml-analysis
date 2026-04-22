import subprocess          # Used to run Linux commands (iw, tshark)
import pandas as pd       # For handling tabular data (DataFrames)
import datetime           # For timestamp processing
import os                 # For file checking
import io                 # To treat string as file (for pandas)
import time               # For delays between channel switching

def set_channel(interface, channel):
    """Tells the Linux OS to change the Wi-Fi card's physical channel."""
    try:
        # Suppress the standard output and error to keep the console clean
        subprocess.run(
            ["iw", "dev", interface, "set", "channel", str(channel)], # Hide output
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL ## Hide errors
        )
        print(f"\n[*] Tuned radio to Channel {channel}...")
        return True
    except subprocess.CalledProcessError:
        print(f"\n[!] Error: Could not switch to Channel {channel}. Skipping...")
        return False

def get_monitor_mode_metrics(location="Unknown", duration=10, interface="wlp0s20f3mon"):
    """
    Captures Wi-Fi packets using tshark in monitor mode and extracts useful metrics.
    """

    # Tshark command to capture packets
    tshark_cmd = [
        "tshark", 
        "-i", interface,                      # Interface in monitor mode
        "-a", f"duration:{duration}",         # Capture duration
        
        # Filter: Only DATA frames (ignore control + beacon noise)
        "-Y", "wlan.fc.type == 2 and wlan.ra[0]&1 == 0", 
        
        "-T", "fields",                       # Output as fields (not raw packets)

        # Fields to extract:
        "-e", "frame.time_epoch",             # Timestamp
        "-e", "wlan.ta",                      # Transmitter MAC
        "-e", "wlan.ra",                      # Receiver MAC
        "-e", "wlan.bssid",                   # Router MAC
        "-e", "wlan.fc.type_subtype",         # Frame subtype
        "-e", "radiotap.dbm_antsignal",       # Signal strength (dBm)
        "-e", "radiotap.channel.freq",        # Frequency
        "-e", "wlan_radio.phy",               # Wi-Fi generation (4/5/6)
        "-e", "radiotap.mcs.index",           # Wi-Fi 4 MCS
        "-e", "wlan_radio.11ac.mcs",          # Wi-Fi 5 MCS
        "-e", "wlan_radio.11ax.mcs",          # Wi-Fi 6 MCS
        "-e", "wlan_radio.data_rate",         # Data rate (Mbps)

        # Output formatting
        "-E", "header=y",                     # Include column headers
        "-E", "separator=,",                  # CSV format
        "-E", "quote=d",                      # Double-quote fields
        "-E", "occurrence=f"                  # First occurrence only
    ]

    try:
        # Run tshark and capture output as string
        perf_output = subprocess.check_output(
            tshark_cmd, encoding="utf-8", errors="ignore"
        )

        # Convert string output into pandas DataFrame
        df = pd.read_csv(io.StringIO(perf_output))
        
        # Rename columns for clarity
        df.columns = [
            "Timestamp", "Transmitter_MAC", "Receiver_MAC", "BSSID",
            "Frame_Type", "Signal_dBm", "Frequency_MHz", 
            "PHY_Type", "MCS_Legacy", "MCS_WiFi5", "MCS_WiFi6", "Data_Rate_Mbps"
        ]
        
        # Add location tag (useful for dataset labeling)
        df.insert(0, "Location", location)

        # Convert timestamp into readable datetime format
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit='s')

        # Remove rows where signal strength is missing
        df = df.dropna(subset=['Signal_dBm'])
        
        return df

    except Exception as e:
        # Return empty DataFrame if something fails
        return pd.DataFrame()


if __name__ == "__main__":

    # Ask user for location tag
    base_loc = input("Enter base location (e.g., SkyDhaba): ").strip()

    # Monitor mode interface
    interface = "wlp0s20f3mon"
    
    # 2.4 GHz Wi-Fi channels
    target_channels = [1,6,11] 
    
    duration_per_channel = 15   # Capture time per channel (seconds)
    master_file = "final_2ghz.csv"
    
    print(f"\n=== Starting Automated Scan Across {len(target_channels)} Channels ===")
    print(f"Total estimated time: {len(target_channels) * duration_per_channel} seconds.")

    for ch in target_channels:

        # Step 1: Switch to channel
        if set_channel(interface, ch):

            # Tag location with channel info
            loc_tag = f"{base_loc}_CH{ch}"
            
            # Step 2: Capture data
            df = get_monitor_mode_metrics(
                location=loc_tag,
                duration=duration_per_channel,
                interface=interface
            )
            
            # Step 3: Save data
            if not df.empty:
                file_exists = os.path.isfile(master_file)

                # Append to CSV (don’t overwrite)
                df.to_csv(
                    master_file,
                    mode='a',
                    index=False,
                    header=not file_exists
                )
                print(f"    -> Success: Saved {len(df)} data frames.")
            else:
                print(f"    -> Dead Air: No valid unicast data captured.")
                
        # Small delay before switching channels
        time.sleep(0.5)

    print("\n=== Data Collection Complete! ===")
