# https://blog.mturk.com/tutorial-a-beginners-guide-to-crowdsourcing-ml-training-data-with-python-and-mturk-d8df4bdf2977

def get_random_comment():
    pass

def get_html(write=False):
    html_str="""
<!DOCTYPE html>
<html>
<head>
<meta http-equiv='Content-Type' content='text/html; charset=UTF-8'/>
<script type='text/javascript' src='https://s3.amazonaws.com/mturk-public/externalHIT_v1.js'></script>
</head>
<body>
<form name='mturk_form' method='post' id='mturk_form' action='https://www.mturk.com/mturk/externalSubmit'><input type='hidden' value='' name='assignmentId' id='assignmentId'/>

<h2>Please answer the following questions about the Reddit comment below.</h2>

<h3>%(comment)s</h3>

<div>
    Is this person making a statement about same-sex marriage?<br>
    <input type='radio' name='relevance' value='Y'>Yes<br>
    <input type='radio' name='relevance' value='N'>No<br>
    <input type='radio' name='relevance' value='U'>Unsure, or there is not enough information to tell<br>
 
    <br>

    Is this person making an argument for or against same-sex marriage rights?<br>
    <input type='radio' name='valence' value='SS'>Strongly in support<br>
    <input type='radio' name='valence' value='WS'>Weakly in support<br>
    <input type='radio' name='valence' value='N'>Neither in support nor against<br>
    <input type='radio' name='valence' value='WA'>Weakly against<br>
    <input type='radio' name='valence' value='SA'>Strongly against<br>
    <input type='radio' name='valence' value='B'>Both (they are representing both positions)<br>
    <input type='radio' name='valence' value='U'>Unsure, or there is not enough information to tell<br>

    <br>

    Is this person's argument consequentialist or values-based? Please select and describe ONE choice of argument type, and leave the other two text boxes blank.<br>
    <input type='radio' name='argument_type' value='C'>Consequentialist<br>
    What is the consequence of their position they are identifying? 
    <input type='text' name='consequence'><br>

    <input type='radio' name='argument_type' value='V'>Values-based<br>
    What is the value they are appealing to? 
    <input type='text' name='value'><br>

    <input type='radio' name='argument_type' value='V'>Neither consequentialist nor values-based<br>
    Please concisely describe the person's argument. 
    <input type='text' name='other_argument'><br>
 
</div>
<p><input type='submit' id='submitButton' value='Submit' /></p></form>

<script language='Javascript'>turkSetAssignmentID();</script>

</body></html>
"""%{'comment':get_random_comment()} 

    if write:
        with open('questions.html', 'w') as fh:
            fh.write(html_str)

    return html_str

def get_xml(write=True):
    xml_str="""
<HTMLQuestion xmlns="http://mechanicalturk.amazonaws.com/AWSMechanicalTurkDataSchemas/2011-11-11/HTMLQuestion.xsd">
<HTMLContent><![CDATA[

<!-- YOUR HTML BEGINS -->
%(html)s
<!-- YOUR HTML ENDS -->

]]>

</HTMLContent>

<FrameHeight>600</FrameHeight>

</HTMLQuestion>
"""%{'html':get_html()}

    if write:
        with open('questions.xml', 'w') as fh:
            fh.write(xml_str)
