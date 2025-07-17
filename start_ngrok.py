from pyngrok import ngrok, conf

# my ngrok auth token
conf.get_default().auth_token = "2xQbkgqcZK3ztGTPA3HLaPDYvpp_7xvMXbiXKZAwb4BwsymU3"  

# Start an HTTP tunnel on port 8000 (Django default)
public_url = ngrok.connect(8000)

# start up info
print("✅ Ngrok tunnel is running!")
print("🌐 Public URL:", public_url)
# print(f"   {public_url}/twiml-message/?name=John&amount=15000")

# Keep the tunnel open
input("Press Enter to stop the tunnel...\n")
# python start_ngrok.py 
