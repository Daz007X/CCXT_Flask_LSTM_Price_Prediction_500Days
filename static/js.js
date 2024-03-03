function BtnClick() { // quando apertado o botão será acionado essa função

    const num_dates = 3 // pegar o valor de entrada na página HTML
    fetch(`http://127.0.0.1:5000/forecast/${num_dates}`) // faz uma solicitação GET na página

        .then(response => response.json()) // passa os dados da solicitação para JSON
        .then(data => { // pega os dados JSON e passa para a função 'data'
            console.log("Date received from server:", data);

            let messages = [];

            for (const date in data.num_dates) {
                const message = `Data: ${date} --> Expected Value: $${data.num_dates[date]}`;
                messages.push(message);
            } // adicionando os dados do nosso forecast para uma array 'messages'

            const graphDiv = document.getElementById("graphDiv"); // div onde queremos exibir o gráfico
            const graphData = JSON.parse(data.graph); // dados do gráfico
            Plotly.newPlot(graphDiv, graphData); // plotagem do gráfico
        })
        .catch(error => {
            console.error("Fetch error:", error);
        }); // se der algum erro ele apontará no console
}

document.addEventListener("DOMContentLoaded", function() {
});