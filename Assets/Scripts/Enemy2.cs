using System.Collections;
using System.Collections.Generic;
using UnityEngine;
[AddComponentMenu("MyGame/Enemy2")]
public class Enemy2 : MonoBehaviour
{
    public float m_speed=1;
    protected float m_rotSpeed=30;
    protected float m_timer=1.5f;
    protected Transform m_transform;
    // public Transform m_ball;
    public float m_life=10;

    // Start is called before the first frame update
    void Start()
    {
        m_transform=this.transform;
    }

    // Update is called once per frame
    void Update()
    {
        UpdateMove();
    }

    float time=0;
    float shootingTime=1.0f;
    protected void UpdateMove(){
        m_timer-=Time.deltaTime;
        if(m_timer<=0){
            m_timer=3;
            m_rotSpeed=-m_rotSpeed;
        }
        m_transform.Rotate(Vector3.up, -m_rotSpeed*Time.deltaTime, Space.World);
        // m_transform.Translate(new Vector3(-m_speed*Time.deltaTime,0,0));
        time+=Time.deltaTime;
        if(time<4f){
            m_transform.Translate(new Vector3(m_speed*Time.deltaTime,0,0));
            // m_transform.Rotate(Vector3.up, m_rotSpeed*Time.deltaTime, Space.World);
        }else{
            m_transform.Translate(new Vector3(-m_speed*Time.deltaTime,0,0));
            // m_transform.Rotate(Vector3.down, -m_rotSpeed*Time.deltaTime, Space.World);
            if(time>8f){
                time=0;
            }
        }
        // Debug.Log(time);

        shootingTime-=Time.deltaTime;
        if(shootingTime<0f){
            // Instantiate(m_ball, m_transform.position, m_transform.rotation);
            shootingTime=1.5f;
        }
    }

    void OnTriggerEnter(Collider other){
        if(other.tag.CompareTo("Ball")==0){
            Ball ball=other.GetComponent<Ball>();
            if(ball!=null){
                // m_life-=ball.m_power;
                // if(m_life<=0){
                //     Destroy(this.gameObject);
                // }
                 Destroy(this.gameObject);
            }
        }
    }
}
