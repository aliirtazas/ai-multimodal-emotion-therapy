document.getElementById('upload-form')?.addEventListener('submit', function(e) {
    e.preventDefault();
    const popup = document.getElementById('processing-popup');
    popup.style.display = 'block';
  
    const formData = new FormData(this);
  
    fetch('/upload', {
      method: 'POST',
      body: formData
    })
    .then(res => res.json())
    .then(data => {
      window.location.href = data.redirect;
    });
  });
  