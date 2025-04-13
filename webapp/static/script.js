document.getElementById('upload-form').addEventListener('submit', function(e) {
  e.preventDefault();

  const form = e.target;
  const formData = new FormData(form);
  document.getElementById('processing-popup').style.display = 'block';

  fetch(form.action, {
    method: form.method,
    body: formData
  })
  .then(response => response.json())
  .then(data => {
    if (data.redirect) {
      window.location.href = data.redirect;
    }
  })
  .catch(err => {
    console.error("Upload failed", err);
    alert("Upload failed. Please try again.");
  });
});

  