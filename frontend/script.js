const API_URL = "http://127.0.0.1:5000/predict";


document.addEventListener(
    "DOMContentLoaded",
    function () {

        const form =
            document.getElementById(
                "prediction-form"
            );

        const button =
            document.getElementById(
                "predict-btn"
            );


        if (!form) {

            console.error(
                "Prediction form not found"
            );

            return;
        }


        if (!button) {

            console.error(
                "Prediction button not found"
            );

            return;
        }


        form.addEventListener(
            "submit",
            function (event) {

                event.preventDefault();

                makePrediction();

            }
        );


        console.log(
            "Explainable AI frontend loaded successfully."
        );

    }
);



async function makePrediction() {

    const button =
        document.getElementById(
            "predict-btn"
        );


    button.disabled = true;

    button.innerText =
        "Analyzing Employee...";


    try {

        const data = {

            Age:
                document.getElementById(
                    "Age"
                ).value,

            Gender:
                document.getElementById(
                    "Gender"
                ).value,

            MaritalStatus:
                document.getElementById(
                    "MaritalStatus"
                ).value,

            DistanceFromHome:
                document.getElementById(
                    "DistanceFromHome"
                ).value,

            Department:
                document.getElementById(
                    "Department"
                ).value,

            JobRole:
                document.getElementById(
                    "JobRole"
                ).value,

            MonthlyIncome:
                document.getElementById(
                    "MonthlyIncome"
                ).value,

            YearsAtCompany:
                document.getElementById(
                    "YearsAtCompany"
                ).value,

            JobSatisfaction:
                document.getElementById(
                    "JobSatisfaction"
                ).value,

            EnvironmentSatisfaction:
                document.getElementById(
                    "EnvironmentSatisfaction"
                ).value,

            OverTime:
                document.getElementById(
                    "OverTime"
                ).value,

            WorkLifeBalance:
                document.getElementById(
                    "WorkLifeBalance"
                ).value

        };


        console.log(
            "Employee data:",
            data
        );


        const response =
            await fetch(
                API_URL,
                {

                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body:
                        JSON.stringify(data)

                }
            );


        const result =
            await response.json();


        console.log(
            "Backend response:",
            result
        );


        if (!response.ok) {

            throw new Error(
                result.error ||
                "Prediction failed"
            );

        }


        displayResult(result);


    } catch (error) {

        console.error(
            "Prediction error:",
            error
        );


        alert(
            "Prediction failed.\n\n" +
            error.message
        );


    } finally {

        button.disabled = false;

        button.innerText =
            "Predict Attrition Risk";

    }

}



function displayResult(result) {

    const resultBox =
        document.getElementById(
            "prediction-result"
        );


    const probability =
        Number(
            result.attrition_probability
        );


    let message;


    if (
        result.prediction ===
        "Attrition"
    ) {

        message =
            "The model predicts a higher likelihood of employee attrition.";

    } else {

        message =
            "The model predicts a lower likelihood of employee attrition.";

    }


    let explanationsHTML = "";


    if (
        result.explanation &&
        result.explanation.length > 0
    ) {


        result.explanation.forEach(
            function (item) {


                const shapValue =
                    Number(
                        item.shap_value
                    );


                explanationsHTML += `

                    <div class="explanation-item">

                        <div class="explanation-feature">

                            ${formatFeatureName(
                                item.feature
                            )}

                        </div>


                        <div class="explanation-value">

                            SHAP Value:
                            ${shapValue.toFixed(4)}

                        </div>


                        <div class="explanation-contribution">

                            ${item.contribution}

                        </div>

                    </div>

                `;

            }
        );

    }


    resultBox.innerHTML = `

        <h2 class="result-title">

            Prediction Result

        </h2>


        <div class="prediction-name">

            ${result.prediction}

        </div>


        <div class="probability">

            Attrition Probability:

            <strong>
                ${probability.toFixed(2)}%
            </strong>

        </div>


        <div class="result-message">

            ${message}

        </div>


        <hr>


        <h2 class="explanation-title">

            Why did the AI make this prediction?

        </h2>


        <p class="explanation-description">

            SHAP explains how different employee
            features influenced this prediction.

        </p>


        ${explanationsHTML}

    `;


    resultBox.classList.add(
        "show"
    );


    resultBox.scrollIntoView({

        behavior: "smooth",

        block: "start"

    });

}



function formatFeatureName(
    name
) {

    return name

        .replaceAll(
            "_",
            " "
        )

        .replace(
            "BusinessTravel Travel Rarely",
            "Business Travel - Travel Rarely"
        )

        .replace(
            "BusinessTravel Travel Frequently",
            "Business Travel - Travel Frequently"
        )

        .replace(
            "JobRole ",
            "Job Role "
        );

}
