var analize_btn = $("#analize-btn");
var imgload = $("#imgload");
var results = $("#results");
var img_viewer = $("#img-viewer");
var img_seconds = $("#img-seconds");
var img_label = $("#img-label");
var file_select = $("#file-select");

analize_btn.on("click", function (event) {
  if (file_select.prop("value") == "") {
    var popover = new bootstrap.Popover(file_select, {
      title: "Campo obligatorio",
      content: "Debe seleccionar una imagen de una fruta para continuar",
    });
    popover.show();
    return;
  }
  var fdata = new FormData();
  jQuery.each(file_select[0].files, function (i, file) {
    fdata.append(i, file);
  });
  $.ajax({
    url: "/analize",
    type: "post",
    method: "POST",
    data: fdata,
    processData: false,
    contentType: false,
    cache: false,
    success: function (responseValue, status, response) {
      console.log([responseValue, status, response]);
      responseValue = JSON.parse(responseValue);

      results.click();
      img_viewer.attr(
        "src",
        "https://www.dsalud.com/wp-content/uploads/2017/03/LA-PAPAYA.jpg"
      );
      img_label.prop("innerHTML", "papaya");
      img_seconds.prop("innerHTML", "3.5");
    },
    error: function (data) {
      console.log(data);
    },
  });
});

$("#button-container button").on("change", function (event) {
  hiddenBox.show();
});
