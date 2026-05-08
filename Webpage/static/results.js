function downloadReport() {

    const weatherReport = document.getElementById("WeatherReport");

    html2canvas(weatherReport, {
        useCORS: true
    }).then(canvas => {

        canvas.toBlob(function(blob) {

            const url = URL.createObjectURL(blob);

            const link = document.createElement("a");

            link.href = url;
            link.download = "weather-report.png";

            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);

            URL.revokeObjectURL(url);

        }, "image/png");

    });

}