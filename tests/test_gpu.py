import iree.runtime as ireert

print("--- IREE Hardware Discovery (HAL API) ---")

try:
    # In the base runtime, we query the HAL (Hardware Abstraction Layer) directly
    driver_names = ireert.HalDriver.query_runtime_driver_names()
    
    if not driver_names:
        print("No drivers detected. Ensure your GPU drivers (Vulkan/ROCm) are installed.")
    else:
        for name in driver_names:
            print(f"\nDriver Found: {name.upper()}")
            try:
                # Create the driver and look for devices
                driver = ireert.HalDriver.create_driver(name)
                device_infos = driver.query_available_devices()
                
                if not device_infos:
                    print("  └── No physical devices found for this driver.")
                else:
                    for info in device_infos:
                        print(f"  └── Device: {info.name}")
                        # Optional: Print some hardware details
                        # print(f"      ID: {info.device_id}")
            except Exception as e:
                print(f"  └── Failed to initialize driver: {e}")

except AttributeError:
    print("Error: Could not find HalDriver. Your IREE installation might be specialized.")
    print("Available attributes in iree.runtime:", dir(ireert))

print("\n--- Discovery Complete ---")