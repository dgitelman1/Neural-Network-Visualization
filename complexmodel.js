const width = 1000;
const height = 600;
let hiddenLayersValue2 = 2; // Default number of hidden layers 2
let nodesValue2 = [16, 8, 4]; // Default number of nodes
let isInitialSetup2 = true;
let networkGraph2;
let model2;
let past_runs = []
let run_num = 1


import {drawGraph} from './script.js';


window.addEventListener('DOMContentLoaded', (event) => {
    d3.selectAll('#nodesDropdown5').style("display", "none")
    d3.selectAll('#nodesDropdown5l').style("display", "none")
    d3.select('#weightCount').html(`The current number of different weights is ${get_weight_count()}`)
    // Event listener for changes in the number of nodes
    document.getElementById("nodesDropdown3").addEventListener("change", function () {
        nodesValue2[0] = parseInt(this.value);
        d3.select('#weightCount').html(`The current number of different weights is ${get_weight_count()}`)
        draw2();
    })
    document.getElementById("nodesDropdown4").addEventListener("change", function () {
        nodesValue2[1] = parseInt(this.value);
        d3.select('#weightCount').html(`The current number of different weights is ${get_weight_count()}`)
        draw2();
    });

    document.getElementById("nodesDropdown5").addEventListener("change", function () {
        nodesValue2[2] = parseInt(this.value);
        d3.select('#weightCount').html(`The current number of different weights is ${get_weight_count()}`)
        draw2();
    });
    document.getElementById("layersDropdown").addEventListener("change", function () {
        hiddenLayersValue2 = parseInt(this.value);
        d3.select('#weightCount').html(`The current number of different weights is ${get_weight_count()}`)
        if (hiddenLayersValue2==1){
            d3.selectAll('#nodesDropdown4').style("display", "none")
            d3.selectAll('#nodesDropdown5').style("display", "none")
            d3.selectAll('#nodesDropdown4l').style("display", "none")
            d3.selectAll('#nodesDropdown5l').style("display", "none")
        }
        if (hiddenLayersValue2==2){
            d3.selectAll('#nodesDropdown4').style("display", "block")
            d3.selectAll('#nodesDropdown5').style("display", "none")
            d3.selectAll('#nodesDropdown4l').style("display", "block")
            d3.selectAll('#nodesDropdown5l').style("display", "none")
        }
        if (hiddenLayersValue2==3){
            d3.selectAll('#nodesDropdown4').style("display", "block")
            d3.selectAll('#nodesDropdown5').style("display", "block")
            d3.selectAll('#nodesDropdown4l').style("display", "block")
            d3.selectAll('#nodesDropdown5l').style("display", "block")
        }
        draw2();
    });
});


class CustomCallback2 extends tf.Callback {
    onBatchEnd(epoch, logs) {
        let weights;
        weights = model2.layers[0].getWeights()[0].arraySync();
        for (let i = 1; i < model2.layers.length; i++) {
            weights = weights.concat(model2.layers[i].getWeights()[0].arraySync())
        }
        weights = weights.flat()
        d3.select("#neuralNet2").selectAll('.link')
        .style("stroke-width", function (d) {
            console.log(d.value)
            d.value = weights[d.overall]
            return 2*Math.abs(d.value)
        })
        .style("stroke", function (d) {
            if (d.value>=0){
                return "green"
            }else{
                return "red"
            }
        })
    }

}

async function trainModel2(model, inputs, labels) {
    let surface = document.getElementById('plot2');
    const get_parameters = get_weight_count();
    const batchSize = 8;
    const epochs = 25;
    const callbacks = tfvis.show.fitCallbacks(surface, ['loss'], { callbacks: ['onEpochEnd'] });
    const history = await model.fit(inputs, labels,
        { batchSize, epochs, shuffle: true, callbacks: [callbacks, new CustomCallback2()] }
    );
    let new_run = {'loss': history['history']['loss'][epochs-1], 'runnum': run_num, 'parameters': get_parameters}
    past_runs.push(new_run);
    run_num = run_num+1;
    draw_bar();
}

function createComplexModel() {
    // Adjust the model creation based on the selected number of nodes and hidden layers
    let model2 = tf.sequential();
    model2.add(tf.layers.dense({ inputShape: [13], units: nodesValue2[0], useBias: true, activation: 'relu' }));

    // Add hidden layers based on the selected number of nodes
    for (let i = 1; i < hiddenLayersValue2; i++) {
        model2.add(tf.layers.dense({ units: nodesValue2[i], activation: 'relu', useBias: true }));
    }

    // Add output layer
    model2.add(tf.layers.dense({ units: 1, useBias: true }));
    const myOptimizer = tf.train.sgd(.001);
    model2.compile({ loss: 'meanSquaredError', optimizer: myOptimizer });
    return model2;
}

function buildNodeGraph2(hiddenLayersValue2, nodesValue2) {
    let newGraph = {
        "nodes": []
    };

    // Construct input layer
    let newFirstLayer2 = [];
    for (let i = 0; i < 13; i++) {
        let newTempLayer2 = { "label": "i" + i, "layer": 1 };
        newFirstLayer2.push(newTempLayer2);
    }

    // Construct hidden layers
    let hiddenLayers2 = [];
    for (let hiddenLayerLoop = 0; hiddenLayerLoop < hiddenLayersValue2; hiddenLayerLoop++) {
        let newHiddenLayer = [];
        // For the height of this hidden layer
        for (let i = 0; i < nodesValue2[hiddenLayerLoop]; i++) {
            let newTempLayer = { "label": "h" + hiddenLayerLoop + i, "layer": (hiddenLayerLoop + 2) };
            newHiddenLayer.push(newTempLayer);
        }
        hiddenLayers2.push(newHiddenLayer);
    }

    // Construct output layer
    let newOutputLayer = [];
    for (let i = 0; i < 1; i++) {
        let newTempLayer = { "label": "o" + i, "layer": hiddenLayersValue2 + 2 };
        newOutputLayer.push(newTempLayer);
    }

    // Add to newGraph
    let allMiddle = newGraph.nodes.concat.apply([], hiddenLayers2);
    newGraph.nodes = newGraph.nodes.concat(newFirstLayer2, allMiddle, newOutputLayer);

    console.log(newGraph);
    return newGraph;
}

function draw2() {
    if (isInitialSetup2) {
        let svg = d3.select("#neuralNet2").append("svg")
            .attr("width", width)
            .attr("height", height);
        networkGraph2 = buildNodeGraph2(hiddenLayersValue2, nodesValue2); // Pass parameters
        drawGraph(networkGraph2, svg);
        isInitialSetup2 = false;
        svg.selectAll(".link").style("stroke-opacity", .4)
    } else {
        let svg = d3.select("#neuralNet2").select("svg")
        svg.selectAll("*").remove()
        console.log("drawing   " + new Date());
        networkGraph2 = buildNodeGraph2(hiddenLayersValue2, nodesValue2); // Pass parameters
        drawGraph(networkGraph2, svg);
        svg.selectAll(".link").style("stroke-opacity", .4)
    }
    model2 = createComplexModel();
}

function get_weight_count(){
    let base = 13*nodesValue2[0];
    for (let i=1; i<hiddenLayersValue2; i++){
        base = base + (nodesValue2[i]*nodesValue2[i-1])
    }
    return base;
}

function draw_bar(){
    d3.select('#count_tracker')
    .style('display', 'flex')
    d3.select('#count_tracker_heading')
    .style('display', 'block')
    const margin = {top: 10, right: 30, bottom: 90, left: 200};
    const params = past_runs.map(d => d.loss)
    const param_scale = d3.scaleLinear([26, 720], ["yellow", "purple"]) 
    const svg = d3.select("#bar_view")
    svg.selectAll("*").remove()
    let width_d = 1000 - margin.left - margin.right;
    let height_d = 600 - margin.top - margin.bottom;
    
    svg
    .attr("width", width_d + margin.left + margin.right)
    .attr("height", height_d + margin.top + margin.bottom)
    .append("g")
    .attr("transform",
          "translate(" + margin.left + "," + margin.top + ")");

    const x = d3.scaleBand()
        .range([ 0, width_d ])
        .domain(past_runs.map(function(d) { return d.runnum; }))
        .padding(0.15);
        svg.append("g")
        .attr("transform", "translate(0," + height_d + ")")
        .call(d3.axisBottom(x))

    const y = d3.scaleLinear()
        .domain([0, Math.max(...params)])
        .range([ height_d, 0]);
        svg.append("g")
        .attr("class", "myYaxis")
        .call(d3.axisLeft(y));

    let u = svg.selectAll("rect")
        .data(past_runs)
        .join("rect")
        .on('mouseover', (event, d) => {
            const tooltip = d3.select('#tooltip');
            tooltip.transition()
            .duration(200)
            .style('position', 'absolute')
            .style('background-color', 'white')
            .style('padding', '6px')
            .style('border', '1px solid #ccc')
            .style('border-radius', '5px')
            .style('font-size', '15px')
            .style("opacity", .9)

            // Set tooltip content
            tooltip.html(`Run number ${d.runnum} <br> Parameter Count: ${d.parameters} <br> Mean squared error: ${d.loss.toFixed(3)}`)
                .style('left', (event.pageX + 10) + 'px')
                .style('top', (event.pageY - 20) + 'px');
        })
        .on('mousemove', function (event) {
            d3.select('#tooltip')
                .style('left', (event.pageX + 10) + 'px')
                .style('top', (event.pageY - 20) + 'px');
        })
        .on('mouseout', () => {
            d3.select('#tooltip')
            .transition()
            .duration(250)
            .style('opacity', 0);
        })
        .transition()
        .duration(1000)
        .attr("x", d => x(d.runnum))
        .attr("y", d => y(d.loss))
        .attr("width", x.bandwidth())
        .attr("height", d => height_d - y(d.loss))
        .attr("fill", d => param_scale(d.parameters));

}


export {CustomCallback2, trainModel2, createComplexModel, draw2, buildNodeGraph2, hiddenLayersValue2, nodesValue2, networkGraph2, model2};
