(function () {
  var dialog = document.getElementById("lightbox");
  var dialogImg = dialog.querySelector("img");

  function openImage(src, alt) {
    dialogImg.src = src;
    dialogImg.alt = alt || "";
    dialog.showModal();
  }

  document.querySelectorAll(".zoom-button").forEach(function (button) {
    button.addEventListener("click", function () {
      var img = button.querySelector("img");
      openImage(button.getAttribute("data-full") || img.src, img.alt);
    });
  });

  dialog.addEventListener("click", function (event) {
    if (event.target !== dialogImg) dialog.close();
  });

  var copyButton = document.getElementById("copy-bibtex");
  copyButton.addEventListener("click", function () {
    var text = document.getElementById("bibtex").innerText;
    var label = copyButton.textContent;
    function done() {
      copyButton.textContent = "Copied";
      setTimeout(function () { copyButton.textContent = label; }, 1500);
    }
    if (navigator.clipboard) {
      navigator.clipboard.writeText(text).then(done);
      return;
    }
    var area = document.createElement("textarea");
    area.value = text;
    document.body.appendChild(area);
    area.select();
    document.execCommand("copy");
    area.remove();
    done();
  });
})();
