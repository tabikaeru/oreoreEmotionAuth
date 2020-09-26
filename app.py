from flask import Flask, request
import ml
import hashlib
from flask_cors import CORS
from redis import Redis

r = Redis()

app = Flask(__name__)
CORS(app)
vectorizer, clf = ml.classifier(2)


@app.route('/', methods=['get'])
def predictHello():
    text = "ok, that went well",
    res = ml.predict(*text, vectorizer, clf)
    return res


@app.route('/train', methods=['get'])
def train():
    ml.train()
    return 'training is succeeded'


@app.route('/signin', methods=['get'])
def signin():
    uname = request.args.get('uname')
    pw = request.args.get('pw')

    redisPWByte = r.get(uname)
    if redisPWByte is None:
        return 'UserName is not exists'

    redisPWStr = redisPWByte.decode(encoding='utf-8')

    replaced_pw = pw.replace('_', ' ')
    resPW = ml.predict(replaced_pw, vectorizer, clf)

    if resPW != redisPWStr:
        return 'Your PassWord is wrong'

    registSentenceByte = r.get(uname + 'RegistSentence')
    registSentenceStr = registSentenceByte.decode(encoding='utf-8')

    if pw == registSentenceStr:
        return 'You should use not same pass word, which you registered'

    concatResUname = resPW.tostring() + uname.encode('utf-8')

    hash = hashlib.sha256(b'uname' + concatResUname).hexdigest()

    return hash


@app.route('/signup', methods=['get'])
def signUp():
    uname = request.args.get('uname')
    pw = request.args.get('pw')

    redisPW = r.get(uname)

    if redisPW:
        return 'Already SignUp'

    replaced_pw = pw.replace('_', ' ')
    res = ml.predict(replaced_pw, vectorizer, clf)
    r.set(uname, res)
    r.set(uname + 'RegistSentence', pw)
    r.expire(uname, 60)
    return 'SignUp succeeded'


if __name__ == "__main__":
    app.run(debug=True)
