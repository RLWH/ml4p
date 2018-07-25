function AddModelPara(x, y, arr) {

    let NewRow = ($('<div />', {
        class: "form-row",
    })).appendTo('body');

    let NewParamBox = $(AddDropDown("Para-" + y + "-" + x, arr, "4"));

    let NewArg = $("<div />", {
        id: "Arg-" + y + "-" + x,
    });
    //AddDropDown("Arg-" + y + "-" + x, arr, "4"));
    //let NewArg = $(AddDropDown("Arg-" + y + "-" + x, arr, "4"));

    NewRow.append($(NewParamBox));
    NewRow.append($(NewArg));

    return NewRow
}


function AddDropDown(id, arr, width) {

    let param_option = ($('<select />', {
        id: id,
        class: "col-md-" + width + " m-1",
    }));

    $(arr).each(function () {
        param_option.append($("<option>").attr('value', this.val).text(this.text));
    });

    return param_option
}

function AddInput(id){
    let input = $("<input />", {
        type:"text",
        class:"form-control",
        id:id,
        text: "Please enter..."
    });
    return input
}

function AddOption(items) {
    let $d = $('<option>');

    $.each(items, function (i, item) {
        $d.append($('<option>', {
            value: item.value,
            text: item.text
        }));
    });
    return $d
}


function Addmodel(x, arr) {

    let Addjumbotron = ($('<div />', {
        id: "Addmodel-" + x,
        class: "jumbotron model-class"
    })).appendTo('body');

    let Row1 = ($('<div />', {
        class: "row"
    }));

    let ModelLabel = ($('<label />', {
        class: "col-sm-1 col-form-label",
        text: "Model"

    }));


    let NewParamBox = $(AddDropDown("model-" + x, arr, "5"));

    Row1.append($(ModelLabel));

    Row1.append($(NewParamBox));

    let Row2 = $("<div />", {
        class: "row",
        html:
        '<div class="col-md-4">Parameter</div>' +
        '<div class="col-md-6">Arguments</div>'
    });

    let ParaContainer = $("<div />", {
        id: "container-" + x
    });


    let AddButton = $("<button />", {
        id: x,
        type: "button",
        class: "btn btn-primary btnAddParam mt-3",
        text: "Add parameters"
    });

    //row4.append($(AddButton));

    Addjumbotron.append($(Row1));
    Addjumbotron.append($(Row2));
    Addjumbotron.append($(ParaContainer));
    Addjumbotron.append($(AddButton));

    return Addjumbotron
}


$(function () {

    let test = [
        {val: 1, text: 'One'},
        {val: 2, text: 'Two'},
        {val: 3, text: 'Three'}
    ];

    let list = [];
    $('#model-container').append($(Addmodel(0, test)));
    $("#inputModel-0").append(AddOption(test));
    $('#container-0').append($(AddModelPara(0, 0, test)));


    let modeloption = $('.model-option');
    let model_name = [];
    let param = [];
    modeloption.empty();
    modeloption.append('<option selected disabled>Choose Model</option>');

    const url = "../static/select.json";

    $.getJSON(url, function (data) {

        $.each(data.PIPELINE, function (i) {
            model_name[i] = data.PIPELINE[i]['model'];
            param[i] = data.PIPELINE[i]['param'];
    })});

    list[0] = 0;

    $(document).on('click', ".btnAddParam", function () {

        var cri_id = $(this).attr('id');
        list[cri_id] = list[cri_id] + 1;

        $('#container-' + cri_id).append($(AddModelPara(list[cri_id], cri_id, test)));
    });

    $(document).on('click', ".btAddModel", function () {
        let new_id = list.length;
        $('#model-container').append($(Addmodel(new_id, test)));
        $('#container-' + new_id).append($(AddModelPara(0, new_id, test)));
        list[new_id] = 0;
    });

    $(document).on('change', "select[id^='Para']", function () {
        let ResetArg = $("#"+$(this).attr('id').replace("Para", "Arg"));
        ResetArg.html($(AddInput("yoyo")));

    });

    $(document).on('click', ":submit", function () {



    });
});


