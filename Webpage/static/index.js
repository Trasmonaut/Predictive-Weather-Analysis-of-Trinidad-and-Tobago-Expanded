document.addEventListener('DOMContentLoaded', function() {
    var elems = document.querySelectorAll('.sidenav');
    var instances = M.Sidenav.init(elems, {});
});


document.addEventListener('DOMContentLoaded', function() {
    var elems = document.querySelectorAll('select');
    var instances = M.FormSelect.init(elems);
});

document.addEventListener('DOMContentLoaded', function() {
    var elems = document.querySelectorAll('.datepicker');
    var instances = M.Datepicker.init(elems, {
        format: 'dd-mm-yyyy',
        autoClose: true,
        yearRange: 10
    });
});
       
AOS.init();

document.addEventListener('DOMContentLoaded', function() {
    var elems = document.querySelectorAll('.modal');
    var modalInstance = M.Modal.init(elems, {
        opacity: 0.8,
        dismissible: true,
      
    });

    var modalInstance = M.Modal.getInstance(document.getElementById('modal1'));
    modalInstance.open();
  });
