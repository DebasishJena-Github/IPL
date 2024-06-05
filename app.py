from flask import Flask, request, jsonify,render_template
import numpy as np

import pickle
model=pickle.load(open('model.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the home page!"

@app.route('/predict', methods=['POST'])
def predict():
    bat_team = request.form.get('bat_team')
    batsman= request.form.get('batsman')
    bowl_team = request.form.get('bowl_team')
    bowler= request.form.get('bowler')
    overs = float(request.form.get('overs'))
    runs = int(request.form.get('runs'))
    wickets = int(request.form.get('wickets'))
    runs_in_prev_5 = int(request.form.get('runs_in_prev_5'))
    wickets_in_prev_5 = int(request.form.get('wickets_in_prev_5'))

    input_data = preprocess_input(bat_team,batsman, bowl_team,bowler, overs, runs, wickets, runs_in_prev_5, wickets_in_prev_5)

    # Predict using the model
    prediction = model.predict(input_data)
    lower_limit = int(prediction[0] - 10)
    upper_limit = int(prediction[0] + 10)

    return jsonify({'lower_limit': lower_limit, 'upper_limit': upper_limit})


def preprocess_input(bat_team, batsman, bowl_team, bowler, overs, runs, wickets, runs_in_prev_5, wickets_in_prev_5):
    # Convert team names to one-hot encoding or label encoding based on how your model was trained
    teams = ['Mumbai Indians', 'Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals', 'Kings XI Punjab',
             'Royal Challengers Bangalore', 'Delhi Daredevils', 'Sunrisers Hyderabad']

    batting_team_encoded = [1 if team == bat_team else 0 for team in teams]
    bowling_team_encoded = [1 if team == bowl_team else 0 for team in teams]

    bats=['SC Ganguly', 'BB McCullum','RT Ponting','DJ Hussey',
            'Mohammad Hafeez', 'PA Patel','ML Hayden','MEK Hussey',
            'MJ Guptill','JC Buttler','KH Pandya','KJ Abbott',
            'TM Head','KW Richardson','NS Naik','SW Billings',
            'AC Gilchrist','Sunny Singh','RG Sharma','A Symonds',

            'MS Dhoni', 'SK Raina', 'JDP Oram', 'S Badrinath',
            'T Kohli','YK Pathan', 'SR Watson', 'M Kaif',
            'DS Lehmann','RA Jadeja','M Rawat', 'D Salunkhe',
            'SK Warne', 'SK Trivedi', 'BE Hendricks','ST Jayasuriya',
            'DJ Thornely', 'RV Uthappa', 'PR Shah','AM Nayar',

            'Imran Tahir', 'MM Sharma', 'DJ Hooda', 'CH Morris',
            'SS Iyer','SA Abbott', 'AN Ahmed', 'YS Chahal',
            'J Suchith', 'P Negi','RG More', 'Anureet Singh',
            'HH Pandya', 'NM Coulter-Nile','PV Tambe', 'MJ McClenaghan',
            'DJ Muthuswami', 'SN Thakur','SN Khan','PJ Cummins']
    bat_encoded = [1 if bat == batsman else 0 for bat in bats]

    bowls = ['P Kumar', 'Z Khan', 'AA Noffke', 'JH Kallis',
            'SB Joshi','CL White', 'B Lee', 'S Sreesanth',
            'JR Hopes', 'IK Pathan','Bipul Sharma', 'DJ Bravo',
            'S Ladda', 'UT Yadav', 'MC Henriques','R McLaren',
            'J Theron', 'S Narwal', 'Sohail Tanvir', 'RS Bopara',

            'Yuvraj Singh', 'YS Chahal', 'Y Venugopal Rao', 'A Mishra',
            'SP Narine', 'Abdur Razzak', 'RR Powar', 'M Ntini',
            'GJ Maxwell', 'BJ Hodge', 'YA Abdulla', 'PP Chawla',
            'RA Jadeja', 'M Muralitharan', 'TM Dilshan', 'VS Malik',
            'D du Preez','RE van der Merwe', 'DL Vettori', 'R Ashwin']

    bowl_encoded = [1 if bowl == bowler else 0 for bowl in bowls]


    # Create the input array
    input_array = batting_team_encoded + bat_encoded + bowling_team_encoded + bowl_encoded + [overs, runs, wickets, runs_in_prev_5, wickets_in_prev_5]

    return np.array(input_array).reshape(1, -1)


if __name__ == '__main__':
    app.run(debug=True)
