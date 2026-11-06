(() => {
  const maxFileSize = 12 * 1024 * 1024;
  const allowedExtensions = new Set(["jpg", "jpeg", "png", "bmp"]);

  const setNotice = (form, message, kind = "error") => {
    const notice = form.querySelector(".notice");
    notice.textContent = message;
    notice.className = `notice notice-${kind}`;
    notice.hidden = false;
  };

  const clearNotice = (form) => {
    const notice = form.querySelector(".notice");
    notice.hidden = true;
    notice.textContent = "";
  };

  const validateFile = (file) => {
    const extension = file.name.split(".").pop().toLowerCase();
    if (!allowedExtensions.has(extension)) {
      return "Unsupported file type. Please choose a JPG, JPEG, PNG, or BMP image.";
    }
    if (file.size > maxFileSize) {
      return "This image is larger than the 12 MB upload limit.";
    }
    if (file.size === 0) {
      return "The selected file is empty.";
    }
    return "";
  };

  const showPreview = (input, file) => {
    const preview = document.getElementById(input.id.replace("Image", "Preview"));
    if (!preview || !file) return;
    const image = preview.querySelector("img");
    const name = preview.querySelector(".preview-name");
    image.src = URL.createObjectURL(file);
    name.textContent = file.name;
    preview.hidden = false;
    input.closest(".upload-zone").hidden = true;
  };

  const clearPreview = (form) => {
    const preview = form.querySelector(".preview-strip");
    const zone = form.querySelector(".upload-zone");
    const image = preview?.querySelector("img");
    if (image?.src.startsWith("blob:")) URL.revokeObjectURL(image.src);
    if (preview) preview.hidden = true;
    if (zone) zone.hidden = false;
  };

  const setLoading = (form, loading) => {
    const button = form.querySelector('button[type="submit"]');
    button.disabled = loading;
    button.querySelector(".button-label").textContent = loading
      ? "Processing image…"
      : form.dataset.idleLabel;
    button.querySelector(".button-loader").hidden = !loading;
  };

  const makeImage = (source, alt, className = "") => {
    const image = document.createElement("img");
    image.src = source;
    image.alt = alt;
    if (className) image.className = className;
    return image;
  };

  const fillImage = (id, source, alt) => {
    const image = document.getElementById(id);
    if (source) {
      image.src = source;
      image.alt = alt;
      image.hidden = false;
    } else {
      image.removeAttribute("src");
      image.hidden = true;
    }
  };

  const setDownload = (id, url) => {
    const link = document.getElementById(id);
    link.href = url;
    link.hidden = !url;
  };

  const setState = (id, label) => {
    const state = document.getElementById(id);
    if (state) state.textContent = label;
  };

  const appendObjectRows = (target, objects) => {
    target.replaceChildren();
    if (!objects.length) {
      const empty = document.createElement("p");
      empty.className = "no-detections";
      empty.textContent = "The model returned no objects above this confidence threshold.";
      target.append(empty);
      return;
    }
    for (const [index, object] of objects.entries()) {
      const row = document.createElement("div");
      row.className = "object-row";
      const number = document.createElement("span");
      number.className = "object-index";
      number.textContent = String(index + 1).padStart(2, "0");
      const label = document.createElement("strong");
      label.textContent = object.label;
      const confidence = document.createElement("span");
      confidence.className = "object-confidence";
      confidence.textContent = `${(object.confidence * 100).toFixed(1)}% confidence`;
      const box = document.createElement("small");
      box.textContent = `Box: ${object.box.map((coordinate) => Math.round(coordinate)).join(", ")}`;
      row.append(number, label, confidence, box);
      target.append(row);
    }
  };

  const renderEnhancement = (data) => {
    document.getElementById("analysisEmpty").hidden = true;
    document.getElementById("analysisResults").hidden = false;
    setState("analysisState", "PROCESSING COMPLETE");
    fillImage("originalResult", data.original, "Uploaded original image");
    fillImage("processedResult", data.processed, "Enhanced image");
    fillImage("edgeResult", data.edges, "Edge detection output");
    document.getElementById("edgeCard").hidden = !data.edges;
    fillImage("thresholdResult", data.threshold, "Adaptive threshold output");
    fillImage("hogResult", data.hog_visualization, "HOG feature visualization");
    fillImage("histogramResult", data.histogram, "Original and enhanced histograms");
    setDownload("enhancedDownload", data.download);
    document.getElementById("analysisDimensions").textContent =
      `${data.dimensions.width} × ${data.dimensions.height}`;
    document.getElementById("originalBrightness").textContent = data.original_brightness;
    document.getElementById("enhancedBrightness").textContent = data.enhanced_brightness;
    document.getElementById("enhancementTime").textContent = data.processing_time;
    document.getElementById("hogSimilarity").textContent = data.hog_similarity;
    document.getElementById("hogCount").textContent = `${data.hog_feature_count.toLocaleString()} HOG features`;
  };

  const renderDetection = (data) => {
    document.getElementById("detectionEmpty").hidden = true;
    document.getElementById("detectionResults").hidden = false;
    setState("detectionState", "INFERENCE COMPLETE");
    fillImage("detectionImageResult", data.result, "YOLO object detections");
    setDownload("detectionDownload", data.download);
    document.getElementById("objectCount").textContent = data.object_count;
    document.getElementById("detectionDimensions").textContent =
      `${data.dimensions.width} × ${data.dimensions.height}`;
    document.getElementById("detectionTime").textContent = data.processing_time;
    appendObjectRows(document.getElementById("objectList"), data.objects);
  };

  const renderSegmentation = (data) => {
    document.getElementById("segmentationEmpty").hidden = true;
    document.getElementById("segmentationResults").hidden = false;
    setState("segmentationState", "SEGMENTATION COMPLETE");
    fillImage("segOriginal", data.original, "Original image");
    fillImage("segResult", data.result, "Instance segmentation overlay");
    setDownload("segDownload", data.download);
    document.getElementById("segObjectCount").textContent = data.object_count;
    document.getElementById("segTime").textContent = data.processing_time;
    appendObjectRows(document.getElementById("segObjectList"), data.objects);

    const grid = document.getElementById("maskGrid");
    grid.replaceChildren();
    if (!data.masks.length) {
      const empty = document.createElement("p");
      empty.className = "no-detections";
      empty.textContent = "No instance masks were returned for this image.";
      grid.append(empty);
    }
    for (const [index, mask] of data.masks.entries()) {
      const card = document.createElement("article");
      card.className = "mask-card";
      card.append(makeImage(mask.url, `Binary mask for ${mask.label}`));
      const title = document.createElement("strong");
      title.textContent = `${String(index + 1).padStart(2, "0")} · ${mask.label}`;
      const details = document.createElement("small");
      details.textContent = mask.confidence === null
        ? "Instance mask"
        : `${(mask.confidence * 100).toFixed(1)}% confidence`;
      const download = document.createElement("a");
      download.className = "text-link";
      download.href = mask.download;
      download.download = "";
      download.textContent = "Download mask ↧";
      card.append(title, details, download);
      grid.append(card);
    }
  };

  const renderAnalytics = (data) => {
    document.getElementById("analyticsEmpty").hidden = true;
    document.getElementById("analyticsResults").hidden = false;
    setState("analyticsState", "REPORT COMPLETE");
    document.getElementById("analyticsObjects").textContent = data.object_count;
    document.getElementById("analyticsDimensions").textContent =
      `${data.dimensions.width} × ${data.dimensions.height}`;
    document.getElementById("analyticsOriginalBrightness").textContent = data.original_brightness;
    document.getElementById("analyticsEnhancedBrightness").textContent = data.enhanced_brightness;
    document.getElementById("analyticsTime").textContent = data.processing_time;
    document.getElementById("analyticsHog").textContent = data.hog_similarity;
    fillImage("analyticsHistogram", data.histogram, "Histogram comparison");
    fillImage("analyticsOriginal", data.original, "Original image");
    fillImage("analyticsEnhanced", data.enhanced, "Gamma-corrected image");
  };

  const renderers = {
    "/api/enhance": renderEnhancement,
    "/api/detect": renderDetection,
    "/api/segment": renderSegmentation,
    "/api/analytics": renderAnalytics,
  };

  document.querySelectorAll("form[data-api]").forEach((form) => {
    form.dataset.idleLabel = form.querySelector(".button-label").textContent;
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      clearNotice(form);
      if (!form.reportValidity()) return;
      const input = form.querySelector('input[type="file"]');
      const validationError = validateFile(input.files[0]);
      if (validationError) {
        setNotice(form, validationError);
        return;
      }

      setLoading(form, true);
      setState(form.id.replace("Form", "State"), "PROCESSING…");
      try {
        const response = await fetch(form.dataset.api, {
          method: "POST",
          body: new FormData(form),
        });
        const data = await response.json();
        if (!response.ok || !data.success) {
          throw new Error(data.error || `Request failed (${response.status}).`);
        }
        renderers[form.dataset.api](data);
        setNotice(form, "Processing completed successfully.", "success");
      } catch (error) {
        setState(form.id.replace("Form", "State"), "PROCESSING FAILED");
        setNotice(form, error.message || "An unexpected processing error occurred.");
      } finally {
        setLoading(form, false);
      }
    });

    form.addEventListener("reset", () => {
      window.setTimeout(() => {
        clearPreview(form);
        clearNotice(form);
        const results = document.querySelector(`#${form.id.replace("Form", "Results")}`);
        const empty = document.querySelector(`#${form.id.replace("Form", "Empty")}`);
        if (results) results.hidden = true;
        if (empty) empty.hidden = false;
        setState(form.id.replace("Form", "State"), "AWAITING IMAGE");
      }, 0);
    });

    form.querySelector(".remove-file").addEventListener("click", () => {
      const input = form.querySelector('input[type="file"]');
      input.value = "";
      clearPreview(form);
      clearNotice(form);
    });

    const fileInput = form.querySelector('input[type="file"]');
    const dropZone = form.querySelector(".upload-zone");
    fileInput.addEventListener("change", () => {
      const file = fileInput.files[0];
      if (!file) return;
      const validationError = validateFile(file);
      if (validationError) {
        fileInput.value = "";
        clearPreview(form);
        setNotice(form, validationError);
      } else {
        clearNotice(form);
        showPreview(fileInput, file);
      }
    });

    ["dragenter", "dragover"].forEach((eventName) => {
      dropZone.addEventListener(eventName, (event) => {
        event.preventDefault();
        dropZone.classList.add("is-dragging");
      });
    });
    ["dragleave", "drop"].forEach((eventName) => {
      dropZone.addEventListener(eventName, (event) => {
        event.preventDefault();
        dropZone.classList.remove("is-dragging");
      });
    });
    dropZone.addEventListener("drop", (event) => {
      const file = event.dataTransfer.files[0];
      if (!file) return;
      const validationError = validateFile(file);
      if (validationError) {
        setNotice(form, validationError);
        return;
      }
      const transfer = new DataTransfer();
      transfer.items.add(file);
      fileInput.files = transfer.files;
      clearNotice(form);
      showPreview(fileInput, file);
    });
  });

  document.querySelectorAll('input[type="range"]').forEach((range) => {
    const output = document.querySelector(`output[for="${range.id}"]`);
    const update = () => {
      output.value = range.value;
      output.textContent = range.value;
    };
    range.addEventListener("input", update);
    range.addEventListener("change", update);
  });

  const menuToggle = document.querySelector(".menu-toggle");
  const navigation = document.querySelector(".primary-nav");
  menuToggle?.addEventListener("click", () => {
    const expanded = menuToggle.getAttribute("aria-expanded") === "true";
    menuToggle.setAttribute("aria-expanded", String(!expanded));
    navigation.classList.toggle("nav-open", !expanded);
  });
})();
