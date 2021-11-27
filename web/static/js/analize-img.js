var analize_btn = $("#analize-btn");
var imgload = $("#imgload");
var image_result = $("#image-result");
var img_viewer = $("#img-viewer");
var img_prob = $("#img-prob");
var img_label = $("#img-label");
var file_select = $("#file-select");
var select_image = $(".select-image");
var image_loading = $(".image_loading");
var select_again = $(".select-again");

select_image.on("click", function (event) {
  file_select.trigger("click");
});
select_again.on("click", function (event) {
  select_image.fadeIn();
  image_loading.fadeOut();
  image_result.fadeOut();
});

file_select.on("change", function (event) {
  if (file_select.prop("value") != "") {
    select_image.fadeOut();
    image_loading.fadeIn();

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

        var reader = new FileReader();
        reader.readAsDataURL(file_select[0].files[0]);
        reader.onloadend = function () {
          var base64data = reader.result;
          img_viewer.attr("src", base64data);
          file_select.val("");
          select_image.fadeOut();
          image_loading.fadeOut();
          image_result.fadeIn();
        };
        img_label.prop("innerHTML", responseValue.clase);
        img_prob.prop("innerHTML", responseValue.probabilidad);
      },
      error: function (data) {
        console.log(data);
        file_select.val("");
        Swal.fire({
          icon: "error",
          title: "Oops...",
          text: "Ocurrió un error al procesar la imagen",
          footer: "Inténtalo nuevamente",
        });
        select_image.fadeIn();
        image_loading.fadeOut();
        image_result.fadeOut();
      },
    });
  }
});
