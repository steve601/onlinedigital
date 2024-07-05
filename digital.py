from flask import Flask,request,render_template
from source.main_project.pipeline.predict_pipeline import PredicPipeline,UserData

app = Flask(__name__)

@app.route('/')
def homepage():
    return render_template('ditital.html')

@app.route('/classify',methods=['POST'])
def do_prediction():
    user_data = UserData(
        gender=request.form.get('gen'),
        campaignchannel=request.form.get('chan'),
        campaigntype=request.form.get('camp'),
        adspend=request.form.get('ad'),
        clickthroughrate=request.form.get('rate'),
        pagespervisit=request.form.get('page'),
        timeonsite=request.form.get('time'),
        emailopens=request.form.get('email'),
        emailclicks=request.form.get('eclick'),
        previouspurchases=request.form.get('purch'),
        loyaltypoints=request.form.get('points')
    )
    user_df = user_data.get_data_as_df()
    
    predictpipe = PredicPipeline()
    y_hat = predictpipe.predict(user_df)
    
    msg = "The customer will convert" if y_hat == 1 else "The customer won't convert"
    
    return render_template('ditital.html',text = msg)

if __name__ == "__main__":
    app.run(host="0.0.0.0")