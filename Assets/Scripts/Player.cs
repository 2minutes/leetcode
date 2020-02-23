using System.Collections;
using System.Collections.Generic;
using UnityEngine;
[AddComponentMenu("MyGame/Player")]
public class Player : MonoBehaviour
{
	public float m_speed=10;
	protected Transform m_transform;
    public Transform m_ball;
    bool hasBall=true;
    // Start is called before the first frame update
    void Start()
    {
        m_transform=this.transform;
    }

    // Update is called once per frame
    void Update()
    {
        float movev=0;
        float moveh=0;

        if(Input.GetKey(KeyCode.UpArrow)){
        	movev+=m_speed*Time.deltaTime;
        }

        if(Input.GetKey(KeyCode.DownArrow)){
        	movev-=m_speed*Time.deltaTime;
        }
        if(Input.GetKey(KeyCode.LeftArrow)){
        	moveh-=m_speed*Time.deltaTime;
        }
        if(Input.GetKey(KeyCode.RightArrow)){
        	moveh+=m_speed*Time.deltaTime;
        }
        this.m_transform.Translate(new Vector3(moveh, 0, movev));

        if(hasBall){
            if(Input.GetKey(KeyCode.Space)||Input.GetMouseButton(0)){
                Instantiate(m_ball, m_transform.position, m_transform.rotation);
                hasBall=false;
            }
        }
    }

    void OnTriggerEnter(Collider other){
        if(other.tag.CompareTo("EnemyBall")==0){
            EnemyBall enemyBall=other.GetComponent<EnemyBall>();
            if(enemyBall!=null){
                // m_life-=ball.m_power;
                // if(m_life<=0){
                //     Destroy(this.gameObject);
                // }
                 Destroy(this.gameObject);
            }
        }else if(other.tag.CompareTo("Ball")==0){
            hasBall=true;
        }
    }
}
