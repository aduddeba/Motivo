async function getRating() {
  const vehicle = document.getElementById("vehicle").value.trim();

  // ✅ Check if the user entered both make and model
  if (vehicle.split(" ").length < 2) {
    alert("Please enter both the make and model (e.g., 'Honda Civic').");
    return;
  }

  const response = await fetch("http://127.0.0.1:8000/predict", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ make_model: vehicle }),
  });

  if (!response.ok) {
    throw new Error(`Server returned status ${response.status}`);
  }

  // ✅ Only show image when make and model were both provided
  const img = document.getElementById("vehicle-image");
  const query = encodeURIComponent(vehicle + " car");
  const imageUrl = `https://source.unsplash.com/400x300/?Honda Civic`;

  img.src = imageUrl;
  img.style.display = "block";
}
