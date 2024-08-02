from flask import Flask,request,render_template,jsonify
from src.pipelines.prediction_pipeline import CustomData,PredictPipeline


application=Flask(__name__)

app=application

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])

def predict_datapoint():
    if request.method=='GET':
        return render_template('form.html')
    
    else:
        data=CustomData(
            mean_radius=float(request.form.get('mean_radius')),
            mean_texture = float(request.form.get('mean_texture')),
            mean_smoothness = float(request.form.get('mean_smoothness')),
            mean_compactness = float(request.form.get('mean_compactness')),
            mean_concavity = float(request.form.get('mean_concavity')),
            mean_symmetry = float(request.form.get('mean_symmetry')),
            mean_fractal_dimension = float(request.form.get('mean_fractal_dimension')),
            radius_error = float(request.form.get('radius_error')),
            texture_error = float(request.form.get('texture_error')),
            smoothness_error = float(request.form.get('smoothness_error')),
            compactness_error = float(request.form.get('compactness_error')),
            concavity_error = float(request.form.get('concavity_error')),
            concave_points_error = float(request.form.get('concave_points_error')),
            symmetry_error = float(request.form.get('symmetry_error')),
            fractal_dimension_error = float(request.form.get('fractal_dimension_error')),
            worst_smoothness = float(request.form.get('worst_smoothness')),
            worst_compactness = float(request.form.get('worst_compactness')),
            worst_concavity = float(request.form.get('worst_concavity')),
            worst_symmetry = float(request.form.get('worst_symmetry')),
            worst_fractal_dimension = float(request.form.get('worst_fractal_dimension')),
        )
        final_new_data=data.get_data_as_dataframe()
        predict_pipeline=PredictPipeline()
        pred=predict_pipeline.predict(final_new_data)

        results=round(pred[0],2)

        return render_template('form.html',final_result=results)
    

if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True)