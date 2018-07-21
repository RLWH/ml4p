$(function () {
    $("h1").click(function () {
        console.log("hi");
    });

    $("#btAddParam").click(function () {
        var numItems = $('#container').length;
        console.log(numItems);

        var domElement = $('.template').html();
        $('#container').append(domElement);
        console.log(domElement);

        // var New = $('<p>yo</p>');
        // $('#container').after(New);

    });
});