function Addparam(x, y) {
    var a = '<div class="form-row">'+
                '<div class="form-group col-md-4">' +
                    '<select id="locality-dropdown" name="model-option">' +
                    '</select>' +
                    //'<select id="inputParam-' + x + '-' + y +'" class="form-control param">' +
                    //    '<option selected>Choose...</option>' +
                    //    '<option>...</option>' +
                    //'</select>' +
                '</div>' +
                //'<div class="form-group col-md-6">' +
                //    '<input type="password" class="form-control arg" id="inputArg" placeholder="Argument">' +
                //'</div>' +
            '</div>';

    return a
}



function Addmodel(x){
    var b = '<div id="model-'+ x +'" class="jumbotron model-class">' +
                '<div class="form-group row">' +
                    '<label for="inputModel-' + x +'" class="col-sm-1 col-form-label">Model</label>' +
                        '<div class="col-sm-5">' +
                            '<select id="inputModel-' + x +'" class="form-control">' +
                                '<option selected>Choose...</option>' +
                                '<option>...</option>' +
                            '</select>' +
                        '</div>' +
                    '</div>' +

                '<div class="row">' +
                    '<div class="form-group col-md-4">' +
                        '<p>Parameter</p>' +
                        // '<label for="inputParam-' + x + '">Parameter</label>' +
                    '</div>' +
                    '<div class="form-group col-md-6">' +
                        '<p>Arguments</p>' +
                        //'<label for="inputArg-' + x + '">Arguments</label>' +
                    '</div>' +
                '</div>' +

                '<div id="container-' + x +'">' +

                '</div>' +

                '<p class="lead">' +
                    '<button id=' + x  +' type="button" class="btn btn-primary btnAddParam">Add parameters</button>' +
                    //'<button id="btnAddParam-' + x + '" type="button" class="btn btn-primary">Add parameters</button>' +
                '</p>' +
            '</div>';
    return b

    var modeloption = $('.model-option');
    modeloption.empty();
    modeloption.append('<option selected disabled>Choose Model</option>');
    modeloption.append($('<option></option>').attr('value', param[i]).text(name[i]));
    //modeloption.append($('<option></option>').attr('value', "A").text("B");
}


// todo: finish json...

$(function () {

    var list = [];
    $('#model-container').append($(Addmodel(0)));
    $('#container-0').append($(Addparam(0, 0)));


    function init(obj) {
        obj.empty();
        obj.append('<option selected disabled>Choose State/Province</option>');
        obj.prop('selectedIndex', 0);
        return obj
    }

    //var dropdown+ = $('#locality-dropdown');
    //dropdown.empty();
    //dropdown.append('<option selected disabled>Choose State/Province</option>');
    //dropdown.prop('selectedIndex', 0);

    var modeloption = $('.model-option');
    var model_name = [];
    var param = [];
    modeloption.empty();
    modeloption.append('<option selected disabled>Choose Model</option>');
    //dropdown.prop('selectedIndex', 0);

    const url = "../static/select.json";

    $.getJSON(url, function (data) {
        console.log("OH");

        $.each(data.PIPELINE, function (i) {
            console.log(data.PIPELINE[i]);
            model_name[i] = data.PIPELINE[i]['model'];
            param[i] = data.PIPELINE[i]['param'];

            console.log(model_name[i]);
            console.log(param[i]);

            modeloption.append($('<option></option>').attr('value', param[i]).text(name[i]));

            //$.each(data.models[i], function (key, val) {
            // todo: extract the list
            //});
        });
    });


    // Populate dropdown with list of provinces
    //$.getJSON(url, function (data) {
    //  $.each(data, function (key, entry) {
    //    dropdown.append($('<option></option>').attr('value', entry.abbreviation).text(entry.name));
    //  });
    //});

    list[0] = 0;

    $(document).on('click', ".btnAddParam", function () {

        var cri_id = $(this).attr('id');
        console.log(cri_id);

        list[cri_id] = list[cri_id] + 1;
        console.log(list[cri_id]);

        $('#container-' + cri_id).append($(Addparam(list[cri_id], cri_id)));
    });

    $(".btAddModel").click(function () {
        var new_id = list.length;
        $('#model-container').append($(Addmodel(new_id)));
        $('#container-' + new_id).append($(Addparam(0, new_id)));
        console.log(new_id);
        list[new_id] = 0;
    });
});

        // var Addparam = $('.template').html();
        //document.getElementById("container").appendChild($(Addparam(1)));

       // var Addparam = $('.template').html();
       //$('#container').append(Addparam);



        // $( "body" ).append( $newdiv1, [ newdiv2, existingdiv1 ] );


        // var New = $('<p>yo</p>');
        // $('#container').after(New);

